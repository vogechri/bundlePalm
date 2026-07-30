"""Summarize coordinator and C++ worker timing for DRS benchmark runs."""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


LOCAL_TIMING_PATTERN = re.compile(r"LOCAL_SOLVE_TIMING\s+(.*)")
COMPLETION_TIMING_PATTERN = re.compile(r"SOLVE_COMPLETION_TIMING\s+(.*)")
VALUE_PATTERN = re.compile(r"([a-z_]+)=([^\s]+)")
SCENE_PATTERN = re.compile(r"problem-(\d+)-")
COMPONENTS = (
	"jacobian_evaluate",
	"jacobian_conversion",
	"assembly",
	"nesterov",
	"cost_evaluate",
)
COMPLETION_COMPONENTS = (
	"contribution",
	"reduction",
	"reply_pack",
	"serialize",
	"mutex_wait",
	"send",
)
DISPATCH_COMPONENTS = ("request_parse", "update_data", "launch_wait")
NESTEROV_INNER_COMPONENTS = (
	"inverse_landmark_blocks",
	"inverse_camera_blocks",
	"rhs_landmark_multiply",
	"rhs_w_multiply",
	"rhs_vector",
	"initialize",
	"iter_w_transpose",
	"iter_landmark_multiply",
	"iter_w_multiply",
	"iter_camera_multiply",
	"iter_vector",
	"iter_stop",
	"final_w_transpose",
	"final_landmark_multiply",
)


def parse_arguments():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--results", required=True, type=Path)
	parser.add_argument("--log-dir", required=True, type=Path)
	parser.add_argument("--variant")
	parser.add_argument("--output", type=Path)
	return parser.parse_args()


def load_results(path, variant):
	rows = []
	with path.open(encoding="utf-8") as stream:
		for line in stream:
			if not line.strip():
				continue
			row = json.loads(line)
			if variant is None or row.get("variant") == variant:
				rows.append(row)
	if not rows:
		raise RuntimeError("no matching result rows")
	return rows


def scene_id(row):
	match = SCENE_PATTERN.search(Path(row["dataset"]).name)
	if match is None:
		raise RuntimeError(f"cannot identify scene from {row['dataset']}")
	return match.group(1)


def find_worker_log(log_dir, row):
	scene = scene_id(row)
	pattern = (
		f"{row['variant']}_{scene}_k{row['clusters']}_i{row['iterations']}_"
		f"l{row['localSteps']}_t{row['threadsPerCluster']}_worker.log"
	)
	matches = sorted(log_dir.glob(pattern))
	if len(matches) != 1:
		raise RuntimeError(
			f"expected one worker log for scene {scene}, found {len(matches)}"
		)
	return matches[0]


def parse_worker_timings(path):
	records_by_cluster = defaultdict(list)
	completion_by_cluster = defaultdict(list)
	with path.open(encoding="utf-8", errors="replace") as stream:
		for line in stream:
			local_match = LOCAL_TIMING_PATTERN.search(line)
			if local_match is not None:
				values = dict(VALUE_PATTERN.findall(local_match.group(1)))
				if values.get("initial") == "0":
					record = {
						key: float(values[key])
						for key in (*COMPONENTS, "total")
					}
					record.update({
						f"nesterov_inner_{key}": float(
							values.get(f"nesterov_{key}", 0.0)
						)
						for key in NESTEROV_INNER_COMPONENTS
					})
					record["nesterov_inner_total"] = float(
						values.get("nesterov_inner_total", 0.0)
					)
					record["nesterov_inner_calls"] = float(
						values.get("nesterov_calls", 0.0)
					)
					record["nesterov_inner_iterations"] = float(
						values.get("nesterov_iterations", 0.0)
					)
					record["nesterov_inner_edge_visits"] = float(
						values.get("nesterov_iterative_edge_visits", 0.0)
					)
					records_by_cluster[int(values["cluster"])].append(record)
				continue
			completion_match = COMPLETION_TIMING_PATTERN.search(line)
			if completion_match is not None:
				values = dict(VALUE_PATTERN.findall(completion_match.group(1)))
				record = {
					key: float(values.get(key, 0.0))
					for key in (
						*DISPATCH_COMPONENTS, *COMPLETION_COMPONENTS, "total"
					)
				}
				completion_by_cluster[int(values["cluster"])].append(record)
	if not records_by_cluster:
		raise RuntimeError(f"no optimization LOCAL_SOLVE_TIMING rows in {path}")
	oracle_counts = {len(records) for records in records_by_cluster.values()}
	if len(oracle_counts) != 1:
		raise RuntimeError(f"unaligned cluster timing counts in {path}")
	if completion_by_cluster:
		if set(completion_by_cluster) != set(records_by_cluster):
			raise RuntimeError(f"unaligned completion timing clusters in {path}")
		for cluster, records in records_by_cluster.items():
			if len(completion_by_cluster[cluster]) != len(records):
				raise RuntimeError(f"unaligned completion timing counts in {path}")
	return records_by_cluster, completion_by_cluster


def summarize_worker_critical_path(records_by_cluster, completion_by_cluster):
	clusters = sorted(records_by_cluster)
	oracle_count = len(records_by_cluster[clusters[0]])
	summary = {key: 0.0 for key in (*COMPONENTS, "total")}
	summary.update({f"completion_{key}": 0.0 for key in (
		*COMPLETION_COMPONENTS, "total"
	)})
	summary.update({f"dispatch_{key}": 0.0 for key in DISPATCH_COMPONENTS})
	summary.update({
		f"nesterov_inner_{key}": 0.0 for key in NESTEROV_INNER_COMPONENTS
	})
	summary.update({
		"nesterov_inner_total": 0.0,
		"nesterov_inner_calls": 0.0,
		"nesterov_inner_iterations": 0.0,
		"nesterov_inner_edge_visits": 0.0,
	})
	worker_cpu_total = 0.0
	for oracle in range(oracle_count):
		records = [records_by_cluster[cluster][oracle] for cluster in clusters]
		critical_index = max(
			range(len(records)), key=lambda index: records[index]["total"]
		)
		if completion_by_cluster:
			completions = [
				completion_by_cluster[cluster][oracle] for cluster in clusters
			]
			completion = completions[critical_index]
		else:
			completion = None
		critical = records[critical_index]
		for key in summary:
			if not key.startswith(("completion_", "dispatch_")):
				summary[key] += critical[key]
		if completion is not None:
			for key in COMPLETION_COMPONENTS:
				summary[f"completion_{key}"] += completion[key]
			summary["completion_total"] += completion["total"]
			for key in DISPATCH_COMPONENTS:
				summary[f"dispatch_{key}"] += completion[key]
		worker_cpu_total += sum(record["total"] for record in records)
	summary["worker_cpu_total"] = worker_cpu_total
	summary["oracle_count"] = oracle_count
	return summary


def summarize_row(row, log_dir):
	worker_timings = parse_worker_timings(find_worker_log(log_dir, row))
	worker = summarize_worker_critical_path(*worker_timings)
	operations = row["optimizationWorkerOperationSeconds"]
	optimization = float(row["optimizationSeconds"])
	solve_batch = float(operations["solveBatch"])
	consensus = float(row["consensusProjectionSeconds"])
	other_operations = sum(
		float(seconds)
		for name, seconds in operations.items()
		if name != "solveBatch"
	)
	dispatch_critical = sum(
		worker[f"dispatch_{key}"] for key in DISPATCH_COMPONENTS
	)
	worker_critical = (
		dispatch_critical + worker["total"] + worker["completion_total"]
	)
	solve_overhead = max(0.0, solve_batch - worker_critical)
	transport = row.get("optimizationTransportPhaseSeconds", {})
	remainder = max(
		0.0, optimization - solve_batch - consensus - other_operations
	)
	return {
		"scene": scene_id(row),
		"optimization": optimization,
		"solve_batch": solve_batch,
		"local_critical": worker["total"],
		"completion_critical": worker["completion_total"],
		"dispatch_critical": dispatch_critical,
		"worker_critical": worker_critical,
		"solve_overhead": solve_overhead,
		"consensus": consensus,
		"objective_control": other_operations,
		"coordinator_remainder": remainder,
		"worker_cpu_total": worker["worker_cpu_total"],
		"oracle_count": worker["oracle_count"],
		**{f"local_{key}": worker[key] for key in COMPONENTS},
		**{
			f"completion_{key}": worker[f"completion_{key}"]
			for key in COMPLETION_COMPONENTS
		},
		**{
			f"dispatch_{key}": worker[f"dispatch_{key}"]
			for key in DISPATCH_COMPONENTS
		},
		**{
			f"nesterov_inner_{key}": worker[f"nesterov_inner_{key}"]
			for key in NESTEROV_INNER_COMPONENTS
		},
		"nesterov_inner_total": worker["nesterov_inner_total"],
		"nesterov_inner_calls": worker["nesterov_inner_calls"],
		"nesterov_inner_iterations": worker["nesterov_inner_iterations"],
		"nesterov_inner_edge_visits": worker["nesterov_inner_edge_visits"],
		**{
			f"transport_{key}": float(transport.get(key, 0.0))
			for key in (
				"batchSetup", "requestBuild", "requestSerialize", "requestSend",
				"replySetup", "replyReceiveWait", "replyParse", "replyDecode",
				"batchFinalize",
			)
		},
	}


def percentage(value, total):
	return 100.0 * value / total if total else 0.0


def render_report(rows, source):
	keys = (
		"optimization",
		"solve_batch",
		"dispatch_critical",
		"local_critical",
		"completion_critical",
		"worker_critical",
		"solve_overhead",
		"consensus",
		"objective_control",
		"coordinator_remainder",
		"worker_cpu_total",
		*tuple(f"local_{key}" for key in COMPONENTS),
		*tuple(f"completion_{key}" for key in COMPLETION_COMPONENTS),
		*tuple(f"dispatch_{key}" for key in DISPATCH_COMPONENTS),
		*tuple(f"nesterov_inner_{key}" for key in NESTEROV_INNER_COMPONENTS),
		"nesterov_inner_total",
		"nesterov_inner_calls",
		"nesterov_inner_iterations",
		"nesterov_inner_edge_visits",
		"transport_requestSerialize",
		"transport_batchSetup",
		"transport_requestBuild",
		"transport_requestSend",
		"transport_replySetup",
		"transport_replyReceiveWait",
		"transport_replyParse",
		"transport_replyDecode",
		"transport_batchFinalize",
	)
	aggregate = {key: sum(row[key] for row in rows) for key in keys}
	profile_name = "ten-scene" if len(rows) == 10 else f"{len(rows)}-scene"
	lines = [
		f"# DRS {profile_name} timing profile",
		"",
		f"Source: `{source}`",
		"",
		"`Worker critical` selects the slowest local-solve cluster for each "
		"oracle and adds only that cluster's completion work. This preserves the "
		"original local critical-path definition without counting another "
		"cluster's barrier wait on top of the slowest solve. Python receive wait "
		"overlaps worker execution and is reported separately rather than added.",
		"",
		"| Scene | Optimization | Dispatch | Local critical | Completion | Solve overhead | Consensus | Objective/control | Other coordinator |",
		"|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
	]
	for row in rows:
		lines.append(
			f"| {row['scene']} | {row['optimization']:.3f}s | "
			f"{row['dispatch_critical']:.3f}s "
			f"({percentage(row['dispatch_critical'], row['optimization']):.1f}%) | "
			f"{row['local_critical']:.3f}s "
			f"({percentage(row['local_critical'], row['optimization']):.1f}%) | "
			f"{row['completion_critical']:.3f}s "
			f"({percentage(row['completion_critical'], row['optimization']):.1f}%) | "
			f"{row['solve_overhead']:.3f}s "
			f"({percentage(row['solve_overhead'], row['optimization']):.1f}%) | "
			f"{row['consensus']:.3f}s "
			f"({percentage(row['consensus'], row['optimization']):.1f}%) | "
			f"{row['objective_control']:.3f}s "
			f"({percentage(row['objective_control'], row['optimization']):.1f}%) | "
			f"{row['coordinator_remainder']:.3f}s "
			f"({percentage(row['coordinator_remainder'], row['optimization']):.1f}%) |"
		)
	total = aggregate["optimization"]
	lines.extend([
		f"| **Total** | **{total:.3f}s** | "
		f"**{aggregate['dispatch_critical']:.3f}s "
		f"({percentage(aggregate['dispatch_critical'], total):.1f}%)** | "
		f"**{aggregate['local_critical']:.3f}s "
		f"({percentage(aggregate['local_critical'], total):.1f}%)** | "
		f"**{aggregate['completion_critical']:.3f}s "
		f"({percentage(aggregate['completion_critical'], total):.1f}%)** | "
		f"**{aggregate['solve_overhead']:.3f}s "
		f"({percentage(aggregate['solve_overhead'], total):.1f}%)** | "
		f"**{aggregate['consensus']:.3f}s "
		f"({percentage(aggregate['consensus'], total):.1f}%)** | "
		f"**{aggregate['objective_control']:.3f}s "
		f"({percentage(aggregate['objective_control'], total):.1f}%)** | "
		f"**{aggregate['coordinator_remainder']:.3f}s "
		f"({percentage(aggregate['coordinator_remainder'], total):.1f}%)** |",
		"",
		"## Local-solve critical path",
		"",
		"| Component | Seconds | Share of local critical path |",
		"|---|---:|---:|",
	])
	for key in COMPONENTS:
		value = aggregate[f"local_{key}"]
		lines.append(
			f"| {key.replace('_', ' ')} | {value:.3f} | "
			f"{percentage(value, aggregate['local_critical']):.1f}% |"
		)
	inner_component_total = sum(
		aggregate[f"nesterov_inner_{key}"]
		for key in NESTEROV_INNER_COMPONENTS
	)
	inner_unattributed = max(
		0.0, aggregate["local_nesterov"] - inner_component_total
	)
	inner_iterations = aggregate["nesterov_inner_iterations"]
	iterative_edge_visits = aggregate["nesterov_inner_edge_visits"]
	iterative_w_seconds = sum(
		aggregate[f"nesterov_inner_{key}"]
		for key in ("iter_w_transpose", "iter_w_multiply")
	)
	edge_payload_gb = iterative_edge_visits * 27 * 8 / 1e9
	lines.extend([
		"",
		"## Inner Nesterov kernel breakdown",
		"",
		f"Critical-path calls: {aggregate['nesterov_inner_calls']:.0f}; "
		f"completed inner iterations: {inner_iterations:.0f}; "
		f"mean iterations per call: "
		f"{aggregate['nesterov_inner_iterations'] / aggregate['nesterov_inner_calls'] if aggregate['nesterov_inner_calls'] else 0.0:.2f}.",
		f"Iterative edge-block visits: {iterative_edge_visits / 1e6:.3f}M; "
		f"27-double edge payload read: {edge_payload_gb:.3f} GB; "
		f"lower-bound edge-payload bandwidth: "
		f"{edge_payload_gb / iterative_w_seconds if iterative_w_seconds else 0.0:.2f} GB/s.",
		"",
		"| Kernel | Seconds | Share of Nesterov | Mean per inner iteration |",
		"|---|---:|---:|---:|",
	])
	for key in NESTEROV_INNER_COMPONENTS:
		value = aggregate[f"nesterov_inner_{key}"]
		per_iteration_us = (
			1e6 * value / inner_iterations if inner_iterations else 0.0
		)
		lines.append(
			f"| {key.replace('_', ' ')} | {value:.3f} | "
			f"{percentage(value, aggregate['local_nesterov']):.1f}% | "
			f"{per_iteration_us:.2f} µs |"
		)
	lines.append(
		f"| function-entry copies, returns, and timer residual | "
		f"{inner_unattributed:.3f} | "
		f"{percentage(inner_unattributed, aggregate['local_nesterov']):.1f}% | "
		f"{1e6 * inner_unattributed / inner_iterations if inner_iterations else 0.0:.2f} µs |"
	)
	iterative_w_share = percentage(
		iterative_w_seconds, aggregate["local_nesterov"]
	)
	transpose_seconds = aggregate["nesterov_inner_iter_w_transpose"]
	forward_seconds = aggregate["nesterov_inner_iter_w_multiply"]
	inverse_seconds = sum(
		aggregate[f"nesterov_inner_{key}"]
		for key in ("inverse_landmark_blocks", "inverse_camera_blocks")
	)
	vector_stop_seconds = sum(
		aggregate[f"nesterov_inner_{key}"]
		for key in ("iter_vector", "iter_stop")
	)
	lines.extend([
		"",
		"### Inner Nesterov interpretation",
		"",
		f"- The iterative `W^T x` and `W y` traversals consume "
		f"{iterative_w_seconds:.3f}s ({iterative_w_share:.1f}% of inner "
		f"Nesterov and {percentage(iterative_w_seconds, total):.1f}% of total "
		"optimization time).",
		f"- `W^T x` takes {transpose_seconds:.3f}s versus "
		f"{forward_seconds:.3f}s for `W y` "
		f"({percentage(transpose_seconds - forward_seconds, forward_seconds):.1f}% "
		"more). The current edge array is camera-major: forward multiplication "
		"updates contiguous camera blocks, while transpose multiplication "
		"scatters into landmark blocks.",
		f"- The two block inversions total {inverse_seconds:.3f}s "
		f"({percentage(inverse_seconds, aggregate['local_nesterov']):.1f}% of "
		"Nesterov); vector acceleration and stopping checks total only "
		f"{vector_stop_seconds:.3f}s "
		f"({percentage(vector_stop_seconds, aggregate['local_nesterov']):.1f}%).",
		f"- The {edge_payload_gb:.3f} GB payload figure excludes vector traffic, "
		"indices, allocation, and cache effects, so its "
		f"{edge_payload_gb / iterative_w_seconds if iterative_w_seconds else 0.0:.2f} "
		"GB/s is a lower-bound effective rate, not measured DRAM bandwidth.",
		"- The first implementation experiment should preserve camera-major "
		"storage for `W y` while adding a landmark-major transpose view (or a "
		"fused landmark-group Schur application) for `W^T x`; optimize iteration "
		"caps only after measuring that layout change.",
	])
	lines.extend([
		"",
		"## C++ pre-solve dispatch critical path",
		"",
		"| Component | Seconds | Share of dispatch critical path |",
		"|---|---:|---:|",
	])
	for key in DISPATCH_COMPONENTS:
		value = aggregate[f"dispatch_{key}"]
		lines.append(
			f"| {key.replace('_', ' ')} | {value:.3f} | "
			f"{percentage(value, aggregate['dispatch_critical']):.1f}% |"
		)
	lines.extend([
		"",
		"## C++ solve completion critical path",
		"",
		"| Component | Seconds | Share of completion critical path |",
		"|---|---:|---:|",
	])
	for key in COMPLETION_COMPONENTS:
		value = aggregate[f"completion_{key}"]
		lines.append(
			f"| {key.replace('_', ' ')} | {value:.3f} | "
			f"{percentage(value, aggregate['completion_critical']):.1f}% |"
		)
	lines.extend([
		"",
		"## Python solve-batch transport",
		"",
		"| Component | Seconds | Share of solve-batch wall time |",
		"|---|---:|---:|",
	])
	for key in (
		"batchSetup", "requestBuild", "requestSerialize", "requestSend",
		"replySetup", "replyReceiveWait", "replyParse", "replyDecode",
		"batchFinalize",
	):
		value = aggregate[f"transport_{key}"]
		lines.append(
			f"| {key} | {value:.3f} | "
			f"{percentage(value, aggregate['solve_batch']):.1f}% |"
		)
	lines.extend([
		"",
		f"Worker CPU-time sum across parallel clusters: "
		f"{aggregate['worker_cpu_total']:.3f}s.",
		"",
	])
	return "\n".join(lines)


def main():
	arguments = parse_arguments()
	rows = [
		summarize_row(row, arguments.log_dir)
		for row in load_results(arguments.results, arguments.variant)
	]
	report = render_report(rows, arguments.results)
	if arguments.output is not None:
		arguments.output.parent.mkdir(parents=True, exist_ok=True)
		arguments.output.write_text(report, encoding="utf-8")
	print(report)


if __name__ == "__main__":
	main()

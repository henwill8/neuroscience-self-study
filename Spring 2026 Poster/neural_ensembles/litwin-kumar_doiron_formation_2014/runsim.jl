#this file is part of litwin-kumar_doiron_formation_2014
#Copyright (C) 2014 Ashok Litwin-Kumar
#see README for more information

using Statistics
using HDF5
#uncomment the line below and set doplot=true to plot a raster
using PyPlot
doplot = true

# Full protocol: 10 s initial + 20 blocks × (20 × (1 s stim + 3 s gap)) + 120 s spontaneous ≈ 1740 s model time.
# Set to `false` to load and test the trained network on spontaneous activity.
const TRAIN_NETWORK = true

include("sim.jl")

# Network size configuration.
const NE_TOTAL = 2000
const NI_TOTAL = 500

# Sparse random mask for all synaptic blocks uses Bernoulli(p) per entry (see simnew).
# If true, p is chosen so expected in-degree per neuron ≈ REFERENCE_SPARSE_P * (Ncells_ref - 1)
# when the network had REFERENCE_NE_SPARSE + REFERENCE_NI_SPARSE cells (default 4000+1000 @ p=0.2).
# Set EXPLICIT_SPARSE_P to a Float64 to force p and ignore matching.
const MATCH_SPARSE_INDEGREE_TO_REFERENCE = false
const REFERENCE_NE_SPARSE = 4000
const REFERENCE_NI_SPARSE = 1000
const REFERENCE_SPARSE_P = 0.2
const EXPLICIT_SPARSE_P = 0.2

# Assembly configuration knobs for sweeps.
# Probability a neuron is in at least one assembly:
#   P(any) = 1 - (1 - p)^N_ASSEMBLIES
# Set assembly size by either:
# - :probability  -> use ASSEMBLY_MEMBERSHIP_PROB directly (or infer from TARGET_ANY_ASSEMBLY_PERCENT)
# - :count        -> use a fixed expected members/assembly count via ASSEMBLY_SIZE_COUNT, with p=count/NE_TOTAL
const ASSEMBLY_SIZE_MODE = :count
const N_ASSEMBLIES = 2
const TARGET_ANY_ASSEMBLY_PERCENT = 64.0
const ASSEMBLY_MEMBERSHIP_PROB = 0.05
const ASSEMBLY_SIZE_COUNT = 200

# Protocol matching mode when TRAIN_NETWORK=true.
# :none                  -> no matching; fixed N_BLOCKS_AT_REFERENCE and N_patterns = N_ASSEMBLIES
# :total_time            -> scales N_blocks ~ 1/N_ASSEMBLIES (keeps total simulated training time near reference)
# :per_assembly_cadence  -> keeps block duration fixed to reference (constant revisit interval per assembly)
#                           by setting N_patterns = REFERENCE_N_ASSEMBLIES_FOR_PROTOCOL and N_blocks fixed.
const MATCH_MODE = :none
const REFERENCE_N_ASSEMBLIES_FOR_PROTOCOL = 20
const N_BLOCKS_AT_REFERENCE = 20

if TRAIN_NETWORK
	if MATCH_MODE == :none
		N_patterns = N_ASSEMBLIES
		N_blocks = N_BLOCKS_AT_REFERENCE
	elseif MATCH_MODE == :total_time
		N_patterns = N_ASSEMBLIES
		N_blocks = max(1, round(Int, N_BLOCKS_AT_REFERENCE * (REFERENCE_N_ASSEMBLIES_FOR_PROTOCOL / N_ASSEMBLIES)))
	elseif MATCH_MODE == :per_assembly_cadence
		N_patterns = REFERENCE_N_ASSEMBLIES_FOR_PROTOCOL
		N_blocks = N_BLOCKS_AT_REFERENCE
	else
		error("Unknown MATCH_MODE=$MATCH_MODE. Use :none, :total_time, or :per_assembly_cadence")
	end
	post_train_ms = 0_000.0
else
	N_blocks = 0
	N_patterns = N_ASSEMBLIES
	post_train_ms = 20_000.0
end

# --- Probe trained network: brief drive to first N members of one assembly (subset column 5 in stim) ---
const KICKSTART_ENABLE = true
const KICKSTART_ASSEMBLY = 1
const KICKSTART_N_SEED_NEURONS = 50
const KICKSTART_T0_MS = 5_500.0
const KICKSTART_DURATION_MS = 500.0
const KICKSTART_DELTA_KHZ = 8.0
const KICKSTART_POST_MS = 5_000.0
const KICKSTART_FREEZE_PLASTICITY = true

function resolve_membership_prob(
	n_assemblies::Int,
	ne_total::Int,
	size_mode::Symbol,
	explicit_p,
	target_any_percent::Float64,
	assembly_size_count::Int,
)
	n_assemblies > 0 || error("N_ASSEMBLIES must be > 0")
	ne_total > 0 || error("NE_TOTAL must be > 0")
	if size_mode == :count
		return clamp(assembly_size_count / ne_total, 0.0, 1.0)
	elseif size_mode == :probability
		if explicit_p !== nothing
			return Float64(explicit_p)
		end
		pany = clamp(target_any_percent / 100.0, 0.0, 1.0)
		return 1.0 - (1.0 - pany)^(1.0 / n_assemblies)
	else
		error("Unknown ASSEMBLY_SIZE_MODE=$size_mode. Use :probability or :count")
	end
end

"""
Training stimulus schedule (times in ms, matching sim.jl).
Columns: population index (1..N_active_assemblies), start, stop, extra Poisson rate (kHz) on targeted E cells.
Optional column 5: if >0, only the first N entries in `popmembers[population,:]` receive the extra drive (subset kickstart).
Baseline external drive remains rex/rix on all cells; stim adds `delta_khz` only to those targeted E cells.
"""
function build_training_stim(;
	N_blocks::Int=20,
	N_patterns::Int=20,
	N_active_assemblies::Int=20,
	initial_transient_ms::Float64=10_000.0,
	stim_duration_ms::Float64=1_000.0,
	gap_duration_ms::Float64=3_000.0,
	delta_khz::Float64=8.0,
)
	rows = N_blocks * N_patterns
	stim = zeros(rows, 4)
	row = 0
	pattern_period = stim_duration_ms + gap_duration_ms
	block_period = N_patterns * pattern_period
	for b in 0:(N_blocks - 1)
		for p in 0:(N_patterns - 1)
			row += 1
			stim[row, 1] = mod(p, N_active_assemblies) + 1
			t0 = initial_transient_ms + b * block_period + p * pattern_period
			stim[row, 2] = t0
			stim[row, 3] = t0 + stim_duration_ms
			stim[row, 4] = delta_khz
		end
	end
	stim
end

function total_protocol_T_ms(;
	N_blocks, N_patterns, initial_transient_ms, stim_duration_ms, gap_duration_ms, post_train_ms,
)
	pattern_period = stim_duration_ms + gap_duration_ms
	block_period = N_patterns * pattern_period
	initial_transient_ms + N_blocks * block_period + post_train_ms
end

function save_trained_h5(path::AbstractString, weights, popmembers, assembly_mask, w_hist_t, w_in_asm_hist, w_out_asm_hist)
	h5open(path, "w") do fid
		g = create_group(fid, "data")
		g["weights"] = weights
		g["popmembers"] = popmembers
		g["assembly_mask"] = UInt8.(assembly_mask)
		g["w_hist_t"] = collect(w_hist_t)
		g["w_in_asm_hist"] = w_in_asm_hist
		g["w_out_asm_hist"] = w_out_asm_hist
	end
	println("saved trained network and weight traces to ", path)
end

"""Raster rows 1..Ne for E cells: bottom = lowest assembly index (min k with mask[k,i]), tie-break by neuron id; neurons in no assembly at top of E block."""
function excitatory_raster_row_order(assembly_mask::AbstractMatrix{Bool}, Ne::Int)
	Npop = size(assembly_mask, 1)
	perm = sortperm(1:Ne, by=function (i)
		min_k = typemax(Int)
		for k in 1:Npop
			if assembly_mask[k, i]
				min_k = min(min_k, k)
			end
		end
		if min_k == typemax(Int)
			(typemax(Int), i)
		else
			(min_k, i)
		end
	end)
	y_e = zeros(Int, Ne)
	for pos in 1:Ne
		y_e[perm[pos]] = pos
	end
	return y_e
end

function plot_weight_timeseries(path::AbstractString, w_hist_t, w_in_asm_hist, w_out_asm_hist)
	n = length(w_hist_t)
	n == 0 && return
	t_sec = w_hist_t ./ 1000.0
	Nasm = size(w_in_asm_hist, 2)
	figure(figsize=(12, 5))
	subplot(1, 2, 1)
	for k in 1:Nasm
		plot(t_sec, view(w_in_asm_hist, :, k), linewidth=0.9, alpha=0.85)
	end
	xlabel("Time (s)")
	ylabel("Mean J_EE (pF)")
	title("Within-assembly mean weight (per assembly)")
	grid(true, alpha=0.3)
	subplot(1, 2, 2)
	for k in 1:size(w_out_asm_hist, 2)
		plot(t_sec, view(w_out_asm_hist, :, k), linewidth=0.9, alpha=0.85)
	end
	xlabel("Time (s)")
	ylabel("Mean J_EE (pF)")
	title("Out of assembly: j in k, i not in k (per assembly)")
	grid(true, alpha=0.3)
	tight_layout()
	savefig(path, dpi=150)
	close()
	println("saved weight time series plot to ", path)
end

p_membership = resolve_membership_prob(
	N_ASSEMBLIES,
	NE_TOTAL,
	ASSEMBLY_SIZE_MODE,
	ASSEMBLY_MEMBERSHIP_PROB,
	TARGET_ANY_ASSEMBLY_PERCENT,
	ASSEMBLY_SIZE_COUNT,
)
expected_any = 100.0 * (1.0 - (1.0 - p_membership)^N_ASSEMBLIES)
expected_assembly_size = round(Int, NE_TOTAL * p_membership)
println("assembly config: N_ASSEMBLIES=", N_ASSEMBLIES,
	", membership p=", round(p_membership, digits=4),
	", expected any-assembly %=", round(expected_any, digits=1),
	", expected members/assembly≈", expected_assembly_size,
	", ASSEMBLY_SIZE_MODE=", ASSEMBLY_SIZE_MODE)
println("network config: NE_TOTAL=", NE_TOTAL, ", NI_TOTAL=", NI_TOTAL)

println("Protocol: TRAIN_NETWORK=", TRAIN_NETWORK,
	(TRAIN_NETWORK ? ", MATCH_MODE=$MATCH_MODE, REFERENCE_N_ASSEMBLIES_FOR_PROTOCOL=$REFERENCE_N_ASSEMBLIES_FOR_PROTOCOL" : ""))

if !TRAIN_NETWORK
	fid = h5open("trained.h5", "r")
	popmembers = read(fid["data"]["popmembers"])
	weights = read(fid["data"]["weights"])
	close(fid)

	Ncells_expected = NE_TOTAL + NI_TOTAL
	size(weights, 1) == Ncells_expected && size(weights, 2) == Ncells_expected ||
		error("trained.h5 weights size $(size(weights)) does not match NE_TOTAL+NI_TOTAL=$Ncells_expected; align NE_TOTAL/NI_TOTAL with the saved network")
	Ne = NE_TOTAL
	assembly_mask = assembly_mask_from_popmembers(popmembers, Ne)

	if KICKSTART_ENABLE
		KICKSTART_ASSEMBLY >= 1 && KICKSTART_ASSEMBLY <= size(popmembers, 1) ||
			error("KICKSTART_ASSEMBLY=$KICKSTART_ASSEMBLY out of range; trained popmembers has $(size(popmembers,1)) rows")
		stim = zeros(1, 5)
		stim[1, 1] = KICKSTART_ASSEMBLY
		stim[1, 2] = KICKSTART_T0_MS
		stim[1, 3] = KICKSTART_T0_MS + KICKSTART_DURATION_MS
		stim[1, 4] = KICKSTART_DELTA_KHZ
		stim[1, 5] = KICKSTART_N_SEED_NEURONS
		T_ms = KICKSTART_T0_MS + KICKSTART_DURATION_MS + KICKSTART_POST_MS
		stdpdelay_probe = KICKSTART_FREEZE_PLASTICITY ? T_ms + 1.0 : 10_000.0
		println("Kickstart probe: assembly=", KICKSTART_ASSEMBLY,
			", seed N=", KICKSTART_N_SEED_NEURONS,
			", pulse [", KICKSTART_T0_MS, ", ", KICKSTART_T0_MS + KICKSTART_DURATION_MS, "] ms",
			", +", KICKSTART_DELTA_KHZ, " kHz, T=", T_ms / 1000, " s, freeze_plasticity=", KICKSTART_FREEZE_PLASTICITY)
	else
		size(popmembers, 1) >= N_patterns || error("trained network has fewer assemblies than N_patterns; lower N_ASSEMBLIES")
		stim = build_training_stim(; N_blocks=N_blocks, N_patterns=N_patterns, N_active_assemblies=N_ASSEMBLIES)
		T_ms = total_protocol_T_ms(;
			N_blocks=N_blocks,
			N_patterns=N_patterns,
			initial_transient_ms=10_000.0,
			stim_duration_ms=1_000.0,
			gap_duration_ms=3_000.0,
			post_train_ms=post_train_ms,
		)
		stdpdelay_probe = 10_000.0
	end

	println("Probe: T=", T_ms / 1000, " s simulation time, ", size(stim, 1), " stimulus epochs")

	times, ns, Ne, Ncells, T, w_hist_t, w_in_asm_hist, w_out_asm_hist, weights = sim(
		stim, weights, popmembers;
		assembly_mask=assembly_mask,
		T=T_ms,
		stdpdelay=stdpdelay_probe,
		weight_stats_every_ms=500.0,
		Ne=NE_TOTAL,
		Ni=NI_TOTAL,
	)
else
	stim = build_training_stim(; N_blocks=N_blocks, N_patterns=N_patterns, N_active_assemblies=N_ASSEMBLIES)
	T_ms = total_protocol_T_ms(;
		N_blocks=N_blocks,
		N_patterns=N_patterns,
		initial_transient_ms=10_000.0,
		stim_duration_ms=1_000.0,
		gap_duration_ms=3_000.0,
		post_train_ms=post_train_ms,
	)
	println("Train: T=", T_ms / 1000, " s, ", size(stim, 1), " stimulus epochs, N_blocks=", N_blocks, ", N_patterns=", N_patterns)
	times, ns, popmembers, assembly_mask, Ne, Ncells, T, w_hist_t, w_in_asm_hist, w_out_asm_hist, weights = simnew(
		stim;
		n_assemblies=N_ASSEMBLIES,
		assembly_membership_prob=p_membership,
		Ne=NE_TOTAL,
		Ni=NI_TOTAL,
		T=T_ms,
		stdpdelay=10_000.0,
		weight_stats_every_ms=500.0,
		match_sparse_indegree_to_reference=MATCH_SPARSE_INDEGREE_TO_REFERENCE && EXPLICIT_SPARSE_P === nothing,
		reference_ne_sparse=REFERENCE_NE_SPARSE,
		reference_ni_sparse=REFERENCE_NI_SPARSE,
		reference_sparse_p=REFERENCE_SPARSE_P,
		sparse_p=EXPLICIT_SPARSE_P,
	)
end

println("mean excitatory firing rate: ", mean(1000 * ns[1:Ne] / T), " Hz")
println("mean inhibitory firing rate: ", mean(1000 * ns[(Ne + 1):Ncells] / T), " Hz")

# Membership stats (excitatory): how many assemblies each neuron belongs to
counts = [sum(assembly_mask[:, i]) for i in 1:Ne]
println("assembly membership per E neuron — min:", minimum(counts),
	", max:", maximum(counts),
	", mean:", mean(counts))
println("neurons in 0 assemblies: ", count(iszero, counts), " (≈ ", round(100 * count(iszero, counts) / Ne, digits=1), "%)")

if size(w_in_asm_hist, 1) > 0
	last = size(w_in_asm_hist, 1)
	println("--- last snapshot: per-assembly W_in (col k = assembly k) ---")
	println("  ", w_in_asm_hist[last, :])
	println("--- last snapshot: per-assembly W_out (j in k, i not in k) ---")
	println("  ", w_out_asm_hist[last, :])
end

plot_weight_timeseries("weight_timeseries.png", w_hist_t, w_in_asm_hist, w_out_asm_hist)
if TRAIN_NETWORK
	save_trained_h5("trained.h5", weights, popmembers, assembly_mask, w_hist_t, w_in_asm_hist, w_out_asm_hist)
end

if doplot
	println("creating full-network raster plot (", Ncells, " neurons)")
	y_e_row = excitatory_raster_row_order(assembly_mask, Ne)
	fig_h = max(5.0, min(24.0, Ncells / 200.0))
	figure(figsize=(10.0, fig_h))
	xlim(0, T)
	ylim(0.5, Ncells + 0.5)
	ylabel("Row: E sorted by first assembly (bottom→top), then I (Ne+1…Ncells)")
	xlabel("Time (ms)")
	tight_layout()

	ms = max(0.08, min(0.35, 120.0 / Ncells))
	for cc in 1:Ncells
		nst = min(ns[cc], size(times, 2))
		nst == 0 && continue
		vals = times[cc, 1:nst]
		isempty(vals) && continue
		col = cc <= Ne ? "k" : "0.45"
		y_plot = cc <= Ne ? Float64(y_e_row[cc]) : Float64(cc)
		scatter(vals, fill(y_plot, length(vals)), s=ms, c=col, marker="o", linewidths=0)
	end
	println("done creating plot")
	savefig("output.png", dpi=150)
	close()
end

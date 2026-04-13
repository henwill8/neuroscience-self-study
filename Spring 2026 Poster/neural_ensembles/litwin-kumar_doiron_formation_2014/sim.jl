#this file is part of litwin-kumar_doiron_formation_2014
#Copyright (C) 2014 Ashok Litwin-Kumar
#see README for more information

using Distributions
using Statistics

# Per-neuron spike-time matrix: width scales with T (ms) and is capped for memory.
const SPIKE_BUFFER_HZ_CAP = 80.0
const SPIKE_BUFFER_COL_CAP = 50_000

"""Binary mask of size (N_assemblies, Ne); assembly_mask[k,i] == true iff E neuron i is in assembly k."""
function sample_assembly_mask(Ne::Int, Npop::Int, p::Float64)
	Bool.(rand(Npop, Ne) .< p)
end

"""
Per-assembly E→E weight means (nonzero J_EE only); weights[j,i] is presynaptic j → postsynaptic i.

- `w_in[k]`: mean weight with j, i both in assembly k.
- `w_out[k]`: mean weight with presynaptic j in k and postsynaptic i not in k (k → outside k).
"""
function assembly_weight_per_assembly(weights::AbstractMatrix{Float64}, assembly_mask::AbstractMatrix{Bool}, Ne::Int)
	Npop, Ne2 = size(assembly_mask)
	Ne2 == Ne || error("assembly_mask Ne mismatch")
	w_in = Vector{Float64}(undef, Npop)
	w_out = Vector{Float64}(undef, Npop)
	for k in 1:Npop
		members = findall(@view assembly_mask[k, :])
		if isempty(members)
			w_in[k] = NaN
		else
			s = 0.0
			c = 0
			for i in members
				for j in members
					w = weights[j, i]
					if w > 0.0
						s += w
						c += 1
					end
				end
			end
			w_in[k] = c > 0 ? s / c : NaN
		end
		sx = 0.0
		cx = 0
		for j in 1:Ne
			assembly_mask[k, j] || continue
			for i in 1:Ne
				assembly_mask[k, i] && continue
				w = weights[j, i]
				if w > 0.0
					sx += w
					cx += 1
				end
			end
		end
		w_out[k] = cx > 0 ? sx / cx : NaN
	end
	return w_in, w_out
end

"""Reconstruct (Npop, Ne) mask from popmembers padded with zeros (for loading checkpoints)."""
function assembly_mask_from_popmembers(popmembers::AbstractMatrix{Int}, Ne::Int)
	Npop, Nmax = size(popmembers)
	m = falses(Npop, Ne)
	for k in 1:Npop
		for ii in 1:Nmax
			idx = popmembers[k, ii]
			idx < 1 && break
			1 <= idx <= Ne || error("popmembers index out of range")
			m[k, idx] = true
		end
	end
	m
end

"""Row-major lists of member indices for stimulus routing; padded with 0 (unused slots)."""
function popmembers_from_mask(assembly_mask::AbstractMatrix{Bool})
	Npop, Ne = size(assembly_mask)
	max_sz = 0
	for k in 1:Npop
		max_sz = max(max_sz, count(assembly_mask[k, :]))
	end
	max_sz = max(max_sz, 1)
	popmembers = zeros(Int, Npop, max_sz)
	for k in 1:Npop
		members = findall(@view assembly_mask[k, :])
		popmembers[k, 1:length(members)] .= members
	end
	popmembers
end

function simnew(
	stim;
	assembly_mask=nothing,
	n_assemblies=20,
	assembly_membership_prob=0.05,
	Ne=4000,
	Ni=1000,
	T=1_740_000.0,
	stdpdelay=10_000.0,
	weight_stats_every_ms=10_000.0,
	match_sparse_indegree_to_reference=false,
	reference_ne_sparse=4000,
	reference_ni_sparse=1000,
	reference_sparse_p=0.2,
	sparse_p=nothing,
) #generates new weights and populations with unpotentiated synapses, runs simulation (times in ms)
	println("setting up weights")

	Ne, Ni, jee0, jei0, jie, jii, p = weightpars(
		Ne=Ne, Ni=Ni;
		sparse_p=sparse_p,
		match_sparse_indegree_to_reference=match_sparse_indegree_to_reference,
		reference_ne_sparse=reference_ne_sparse,
		reference_ni_sparse=reference_ni_sparse,
		reference_sparse_p=reference_sparse_p,
	)
	if match_sparse_indegree_to_reference
		Nc = Ne + Ni
		Nref = reference_ne_sparse + reference_ni_sparse
		exp_k = reference_sparse_p * (Nref - 1)
		println("sparse mask: p=", round(p, digits=5),
			" (target E in-degree ≈ ", round(exp_k, digits=1),
			" of ", Nc - 1, " presynaptic slots, matched to reference N=$Nref, p_ref=$reference_sparse_p)")
	elseif sparse_p !== nothing
		println("sparse mask: p=", round(p, digits=5), " (explicit sparse_p)")
	end
	Ncells = Ne+Ni

	Npop = n_assemblies
	pmembership = assembly_membership_prob
	if assembly_mask === nothing
		assembly_mask = sample_assembly_mask(Ne, Npop, pmembership)
	end
	size(assembly_mask) == (Npop, Ne) || error("assembly_mask must be (n_assemblies, Ne)")
	popmembers = popmembers_from_mask(assembly_mask)

	#set up weights
	#note: weights are set up so that w[i,j] is weight from presynaptic i to postsynaptic j
	#this is for performance: iterating over presynaptic indices is more important and
	#Julia uses column-major arrays
	weights = zeros(Ncells,Ncells)
	weights[1:Ne,1:Ne] .= jee0
	weights[1:Ne,(1+Ne):Ncells] .= jie
	weights[(1+Ne):Ncells,1:Ne] .= jei0
	weights[(1+Ne):Ncells,(1+Ne):Ncells] .= jii

	weights = weights.*(rand(Ncells,Ncells) .< p)
	for cc = 1:Ncells
		weights[cc,cc] = 0
	end

	times, ns, Ne, Ncells, T, w_hist_t, w_in_asm_hist, w_out_asm_hist, weights = sim(
		stim, weights, popmembers;
		assembly_mask=assembly_mask,
		T=T,
		stdpdelay=stdpdelay,
		weight_stats_every_ms=weight_stats_every_ms,
		Ne=Ne,
		Ni=Ni,
	)
	return times, ns, popmembers, assembly_mask, Ne, Ncells, T, w_hist_t, w_in_asm_hist, w_out_asm_hist, weights
end


function sim(stim, weights, popmembers;
	assembly_mask=nothing,
	T=2000.0,
	stdpdelay=1000.0,
	weight_stats_every_ms=10000.0,
	Ne=nothing,
	Ni=nothing,
) #runs simulation given weight matrix and populations; times in ms
	println("setting up parameters")

	if Ne === nothing || Ni === nothing
		Ne, Ni, jee0, jei0, jie, jii, p = weightpars()
	else
		Ne, Ni, jee0, jei0, jie, jii, p = weightpars(Ne=Ne, Ni=Ni)
	end
	Ncells = Ne + Ni
	(size(weights, 1), size(weights, 2)) == (Ncells, Ncells) ||
		error("weights must be (Ne+Ni)×(Ne+Ni); got size($(size(weights,1)),$(size(weights,2))) for Ne=$Ne, Ni=$Ni")

	#membrane dynamics
	taue = 20 #e membrane time constant
	taui = 20 #i membrane time constant
	vleake = -70 #e resting potential
	vleaki = -62 #i resting potential
	deltathe = 2 #eif slope parameter
	C = 300 #capacitance
	erev = 0 #e synapse reversal potential
	irev = -75 #i synapse reversal potntial
	vth0 = -52 #initial spike voltage threshold
	ath = 10 #increase in threshold post spike
	tauth = 30 #threshold decay timescale
	vre = -60 #reset potential
	taurefrac = 1 #absolute refractory period
	aw_adapt = 4 #adaptation parameter a
	bw_adapt = .805 #adaptation parameter b
	tauw_adapt = 150 #adaptation timescale

	#connectivity
	tauerise = 1 #e synapse rise time
	tauedecay = 6 #e synapse decay time
	tauirise = .5 #i synapse rise time
	tauidecay = 2 #i synapse decay time
	rex = 4.5 #external input rate to e (khz)
	rix = 2.25 #external input rate to i (khz)

	jeemin = 1.78 #minimum ee strength
	jeemax = 21.4 #maximum ee strength

	jeimin = 48.7 #minimum ei strength
	jeimax = 243 #maximum ei strength

	jex = 1.78 #external to e strength
	jix = 1.27 #external to i strength

	#voltage based stdp
	altd = .0008 #ltd strength
	altp = .0014 #ltp strength
	thetaltd = -70 #ltd voltage threshold
	thetaltp = -49 #ltp voltage threshold
	tauu = 10 #timescale for u variable
	tauv = 7 #timescale for v variable
	taux = 15 #timescale for x variable

	#inhibitory stdp
	tauy = 20 #width of istdp curve
	eta = 1 #istdp learning rate
	r0 = .003 #target rate (khz)

	#populations
	Npop = size(popmembers,1) #number of assemblies
	Nmaxmembers = size(popmembers,2) #maximum number of neurons in a population

	#simulation
	dt = .1 #integration timestep (ms); state time t is in ms
	vpeak = 20 #cutoff for voltage.  when crossed, record a spike and reset
	dtnormalize = 20 #how often to normalize rows of ee weights (ms)
	# Spike-time buffer: ns[c] counts all spikes; times only stores the first Nspikes columns per neuron.
	Nspikes = max(100, min(SPIKE_BUFFER_COL_CAP, ceil(Int, (T / 1000) * SPIKE_BUFFER_HZ_CAP)))
	track_weights = assembly_mask !== nothing
	weight_stats_every = max(1, round(Int, weight_stats_every_ms / dt))

	times = zeros(Ncells,Nspikes)
	ns = zeros(Int,Ncells)

	forwardInputsE = zeros(Ncells) #summed weight of incoming E spikes
	forwardInputsI = zeros(Ncells)
	forwardInputsEPrev = zeros(Ncells) #as above, for previous timestep
	forwardInputsIPrev = zeros(Ncells)

	xerise = zeros(Ncells) #auxiliary variables for E/I currents (difference of exponentials)
	xedecay = zeros(Ncells)
	xirise = zeros(Ncells)
	xidecay = zeros(Ncells)

	expdist = Exponential()

	v = zeros(Ncells) #membrane voltage 
	nextx = zeros(Ncells) #time of next external excitatory input
	sumwee0 = zeros(Ne) #initial summed e weight, for normalization
	Nee = zeros(Int,Ne) #number of e->e inputs, for normalization
	rx = zeros(Ncells) #rate of external input
	for cc = 1:Ncells
		v[cc] = vre + (vth0-vre)*rand()
		if cc <= Ne 
			rx[cc] = rex
			nextx[cc] = rand(expdist)/rx[cc]
			for dd = 1:Ne
				sumwee0[cc] += weights[dd,cc]
				if weights[dd,cc] > 0 
					Nee[cc] += 1
				end
			end
		else
			rx[cc] = rix
			nextx[cc] = rand(expdist)/rx[cc]
		end
	end

	vth = vth0*ones(Ncells) #adaptive threshold
	wadapt = aw_adapt*(vre-vleake)*ones(Ne) #adaptation current
	lastSpike = -100*ones(Ncells) #last time the neuron spiked
	trace_istdp = zeros(Ncells) #low-pass filtered spike train for istdp
	u_vstdp = vre*zeros(Ne)
	v_vstdp = vre*zeros(Ne)
	x_vstdp = zeros(Ne)

	Nsteps = round(Int,T/dt)
	inormalize = round(Int,dtnormalize/dt)
	n_w_rec = track_weights ? div(Nsteps, weight_stats_every) : 0
	w_hist_t = zeros(Float64, n_w_rec)
	w_in_asm_hist = fill(NaN, n_w_rec, Npop)
	w_out_asm_hist = fill(NaN, n_w_rec, Npop)
	i_w_rec = 0

	println("starting simulation")

	#begin main simulation loop
	for tt = 1:Nsteps
		if mod(tt,Nsteps/100) == 1  #print percent complete
			print("\r",round(Int,100*tt/Nsteps))
		end
		t = dt*tt
		forwardInputsE[:] .= 0.
		forwardInputsI[:] .= 0.

		#check if we have entered or exited a stimulation period
		tprev = dt*(tt-1)
		stim_ncol = size(stim, 2)
		for ss = 1:size(stim, 1)
			# Optional column 5: only first N members of assembly (for subset / kickstart); 0 or absent = all members
			subset_n = stim_ncol >= 5 ? max(0, round(Int, stim[ss, 5])) : 0
			ii_max = subset_n > 0 ? min(Nmaxmembers, subset_n) : Nmaxmembers
			if (tprev < stim[ss, 2]) && (t >= stim[ss, 2])  #just entered stimulation period
				ipop = round(Int, stim[ss, 1])
				for ii = 1:ii_max
					if popmembers[ipop, ii] < 1
						break
					end
					rx[popmembers[ipop, ii]] += stim[ss, 4]
				end
			end

			if (tprev < stim[ss, 3]) && (t >= stim[ss, 3]) #just exited stimulation period
				ipop = round(Int, stim[ss, 1])
				for ii = 1:ii_max
					if popmembers[ipop, ii] < 1
						break
					end
					rx[popmembers[ipop, ii]] -= stim[ss, 4]
				end
			end
		end #end loop over stimuli

		if track_weights && mod(tt, weight_stats_every) == 0
			i_w_rec += 1
			if i_w_rec <= n_w_rec
				w_hist_t[i_w_rec] = t
				win, wout = assembly_weight_per_assembly(weights, assembly_mask, Ne)
				w_in_asm_hist[i_w_rec, :] .= win
				w_out_asm_hist[i_w_rec, :] .= wout
			end
		end

		if mod(tt,inormalize) == 0 #excitatory synaptic normalization
			for cc = 1:Ne
				sumwee = 0.
				for dd = 1:Ne
					sumwee += weights[dd,cc]
				end

				for dd = 1:Ne
					if weights[dd,cc] > 0.
						weights[dd,cc] -= (sumwee-sumwee0[cc])/Nee[cc]
						if weights[dd,cc] < jeemin
							weights[dd,cc] = jeemin
						elseif weights[dd,cc] > jeemax
							weights[dd,cc] = jeemax
						end
					end
				end
			end
		end #end normalization

		#update single cells
		spiked = zeros(Bool,Ncells)	
		for cc = 1:Ncells
			trace_istdp[cc] -= dt*trace_istdp[cc]/tauy

			while(t > nextx[cc]) #external input
				nextx[cc] += rand(expdist)/rx[cc]
				if cc < Ne
					forwardInputsEPrev[cc] += jex
				else
					forwardInputsEPrev[cc] += jix
				end
			end

			xerise[cc] += -dt*xerise[cc]/tauerise + forwardInputsEPrev[cc]
			xedecay[cc] += -dt*xedecay[cc]/tauedecay + forwardInputsEPrev[cc]
			xirise[cc] += -dt*xirise[cc]/tauirise + forwardInputsIPrev[cc]
			xidecay[cc] += -dt*xidecay[cc]/tauidecay + forwardInputsIPrev[cc]

			if cc < Ne
				vth[cc] += dt*(vth0 - vth[cc])/tauth;
				wadapt[cc] += dt*(aw_adapt*(v[cc]-vleake) - wadapt[cc])/tauw_adapt;
				u_vstdp[cc] += dt*(v[cc] - u_vstdp[cc])/tauu;
				v_vstdp[cc] += dt*(v[cc] - v_vstdp[cc])/tauv;
				x_vstdp[cc] -= dt*x_vstdp[cc]/taux;
			end

			if t > (lastSpike[cc] + taurefrac) #not in refractory period
				# update membrane voltage
				ge = (xedecay[cc] - xerise[cc])/(tauedecay - tauerise);
				gi = (xidecay[cc] - xirise[cc])/(tauidecay - tauirise);

				if cc < Ne #excitatory neuron (eif), has adaptation
					dv = (vleake - v[cc] + deltathe*exp((v[cc]-vth[cc])/deltathe))/taue + ge*(erev-v[cc])/C + gi*(irev-v[cc])/C - wadapt[cc]/C;
					v[cc] += dt*dv;
					if v[cc] > vpeak
						spiked[cc] = true
						wadapt[cc] += bw_adapt
					end
				else
					dv = (vleaki - v[cc])/taui + ge*(erev-v[cc])/C + gi*(irev-v[cc])/C;
					v[cc] += dt*dv;
					if v[cc] > vth0
						spiked[cc] = true
					end
				end

				if spiked[cc] #spike occurred
					spiked[cc] = true;
					v[cc] = vre;
					lastSpike[cc] = t;
					ns[cc] += 1;
					if ns[cc] <= Nspikes
						times[cc,ns[cc]] = t;
					end
					trace_istdp[cc] += 1.;
					if cc<Ne
						x_vstdp[cc] += 1. / taux;
					end

					if cc < Ne
						vth[cc] = vth0 + ath;
					end
					
					#loop over synaptic projections 
					for dd = 1:Ncells
						if cc <= Ne #excitatory synapse
							forwardInputsE[dd] += weights[cc,dd];
						else #inhibitory synapse
							forwardInputsI[dd] += weights[cc,dd];
						end
					end

				end #end if(spiked)
			end #end if(not refractory)
			
			#istdp
			if spiked[cc] && (t > stdpdelay)
				if cc < Ne #excitatory neuron fired, potentiate i inputs
					for dd = (Ne+1):Ncells
						if weights[dd,cc] == 0.
							continue
						end
						weights[dd,cc] += eta*trace_istdp[dd]
						if weights[dd,cc] > jeimax
							weights[dd,cc] = jeimax
						end
					end	
				else #inhibitory neuron fired, modify outputs to e neurons
					for dd = 1:Ne
						if weights[cc,dd] == 0.
							continue
						end
						weights[cc,dd] += eta*(trace_istdp[dd] - 2*r0*tauy)
						if weights[cc,dd] > jeimax
							weights[cc,dd] = jeimax
						elseif weights[cc,dd] < jeimin
							weights[cc,dd] = jeimin
						end
					end	
				end
			end #end istdp


			#vstdp, ltd component
			if spiked[cc] && (t > stdpdelay) && (cc < Ne)
				for dd = 1:Ne #depress weights from cc to cj
					if weights[cc,dd] == 0.
						continue
					end

					if u_vstdp[dd] > thetaltd
						weights[cc,dd] -= altd*(u_vstdp[dd]-thetaltd)
						if weights[cc,dd] < jeemin
							weights[cc,dd] = jeemin

						end
					end
				end
			end #end ltd

			#vstdp, ltp component
			if (t > stdpdelay) && (cc < Ne) && (v[cc] > thetaltp) && (v_vstdp[cc] > thetaltd)
				for dd = 1:Ne
					if weights[dd,cc] == 0.
						continue
					end

					weights[dd,cc] += dt*altp*x_vstdp[dd]*(v[cc] - thetaltp)*(v_vstdp[cc] - thetaltd);
					if weights[dd,cc] > jeemax
						weights[dd,cc] = jeemax
					end
				end
			end #end ltp

		end #end loop over cells
		forwardInputsEPrev = copy(forwardInputsE)
		forwardInputsIPrev = copy(forwardInputsI)
	end #end loop over time
	print("\r")

	ncol = maximum(min.(ns, Nspikes))
	times = ncol > 0 ? times[:, 1:ncol] : times[:, 1:1]

	return times, ns, Ne, Ncells, T, w_hist_t, w_in_asm_hist, w_out_asm_hist, weights
end

"""Number of spike times actually stored in `times` for one neuron (`ns` may be larger if buffer filled)."""
function n_stored_spike_times(ns_c::Int, times::AbstractMatrix)
	min(ns_c, size(times, 2))
end

function weightpars(;
	Ne=4000,
	Ni=1000,
	sparse_p=nothing,
	match_sparse_indegree_to_reference=false,
	reference_ne_sparse=4000,
	reference_ni_sparse=1000,
	reference_sparse_p=0.2,
) #parameters needed to generate weight matrix
	jee0 = 2.86 #initial ee strength
	jei0 = 48.7 #initial ei strength
	jie = 1.27 #ie strength (not plastic)
	jii = 16.2 #ii strength (not plastic)
	p_default = 0.2
	if sparse_p !== nothing
		p = clamp(Float64(sparse_p), 0.0, 1.0)
	elseif match_sparse_indegree_to_reference
		Nref = reference_ne_sparse + reference_ni_sparse
		Ncur = Ne + Ni
		Nref > 1 || error("reference network must have Ncells > 1 for indegree matching")
		p = reference_sparse_p * (Nref - 1) / max(Ncur - 1, 1)
		p = clamp(p, 0.0, 1.0)
	else
		p = p_default
	end
	return Ne, Ni, jee0, jei0, jie, jii, p
end



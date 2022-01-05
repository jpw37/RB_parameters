from trial_run import DA_simulate

DA = DA_simulate(Rayleigh=5e6, warmup_time=0.5)
DA.run_assimilator(outdir='output',final_sim_time=0.5)

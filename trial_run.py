from RB_2D import *
from RB_2D_assim import *
import numpy as np
from functools import partial
import json
import itertools
import os
import shutil
import sys
from tqdm import tqdm


#@staticmethod
def P_N(F, N, scale=False):
    """Calculate the Fourier mode projection of F with N terms."""
    # Set the c_n to zero wherever n > N (in both axes).
    X,Y = np.indices(F['c'].shape)
    F['c'][(X >= N) | (Y >= N)] = 0

    if scale:
        F.set_scales(1)
    return F['g']

class DA_simulate(object):
    """A class to run data assimilations on the 2D RB system
    """

    def __init__(self, outdir='new_run', L=4, xsize=384, zsize=192, Prandtl=1,
                 Rayleigh=1e6, mu = 1, N=32, warmup_time=2, warmup_dt=1e-5,
                 overwrite=False):
        """Run normal 2D RB for warmup_time to get into the turbulent setting.
        """

        truth = RB_2D(L=L, xsize=xsize, zsize=zsize, Prandtl=Prandtl, Rayleigh=Rayleigh)
        truth.setup_evolution()
        truth.simulate_setup()
        truth.solver.stop_sim_time = warmup_time
        truth.solver.stop_wall_time = 1e10
        truth.solver.stop_iteration = np.inf

        try:
            truth.logger.info("Starting warmup loop")
            start_time = time.time()
            while truth.solver.ok:
                dt = truth.cfl.compute_dt()
                dt = truth.solver.step(dt)

                if truth.solver.iteration % 10 == 0:
                    info = "Iteration {:>5d}, Time: {:.7f}, dt: {:.2e}".format(
                        truth.solver.iteration, truth.solver.sim_time,dt)
                    Re = truth.flow.max("Re")
                    info += ", Max Re = {:f}".format(Re)
                    truth.logger.info(info)
                    if np.isnan(Re):
                        raise ValueError("Reynolds number went to infinity!!"
                                         "\nRe = {}", format(Re))
        except BaseException as e:
            truth.logger.error("Exception raised, triggering end of main loop.")
            raise
        finally:
            total_time = time.time()-start_time
            cpu_hr = total_time/60/60*SIZE
            truth.logger.info("Iterations: {:d}".format(truth.solver.iteration))
            truth.logger.info("Sim end time: {:.3e}".format(truth.solver.sim_time))
            truth.logger.info("Run time: {:.3e} sec".format(total_time))
            truth.logger.info("Run time: {:.3e} cpu-hr".format(cpu_hr))
            truth.logger.debug("END OF SIMULATION")

        assimilator = RB_2D_assim(projector=P_N, mu=mu, N=N, L=L, xsize=xsize,
                                  zsize=zsize, Prandtl=Prandtl, Rayleigh=Rayleigh)
        assimilator.setup_evolution()
        assimilator.simulate_setup()
        assimilator.finalize_solver(truth.solver.state['zeta'])
        self.assimilator = assimilator
        self.truth = truth


    def run_assimilator(self, outdir, final_sim_time=5):
        assimilator = self.assimilator
        truth = self.truth
        assimilator.solver.stop_sim_time = final_sim_time
        assimilator.solver.stop_wall_time = 1e10
#        assimilator.solver.stop_iteration = np.inf
        truth.solver.stop_sim_time += final_sim_time

        try:
            assimilator.logger.info("Starting main loop")
            start_time = time.time()
            while truth.solver.ok & assimilator.solver.ok:
                dt = np.min([truth.cfl.compute_dt(), assimilator.cfl.compute_dt()])
                dt = truth.solver.step(dt)
                dzeta = assimilator.problem.domain.new_field(name='dzeta')
                zeta = truth.solver.state['zeta']
                zeta.set_scales(1)
                zeta_assim = assimilator.solver.state['zeta']
                zeta_assim.set_scales(1)
                dzeta['g'] = zeta_assim['g'] - zeta['g']
#                dzeta['g'] = assimilator.solver.state['zeta']['g'] - truth.solver.state['zeta']['g']
                assimilator.problem.parameters["driving"].args = [dzeta, assimilator.N]
                dt = assimilator.solver.step(dt)

                if truth.solver.iteration % 10 == 0:
                    info = "Iteration {:>5d}, Time: {:.7f}, dt: {:.2e}".format(
                        truth.solver.iteration, truth.solver.sim_time, dt)
                    Re = truth.flow.max("Re")
                    info += ", Max Re = {:f}".format(Re)
                    truth.logger.info(info)
                    info_assim = "Assimilator iteration {:>5d}, Time: {:.7f}, dt: {:.2e}".format(
                        assimilator.solver.iteration, assimilator.solver.sim_time, dt)
                    Re_assim = assimilator.flow.max("Re")
                    info_assim += ", Max Re = {:f}".format(Re_assim)
                    assimilator.logger.info(info_assim)
                    if np.isnan(Re):
                        raise ValueError("Reynolds number went to infinity!!"
                                         "\nRe = {}", format(Re))
        except BaseException as e:
            truth.logger.error("Exception raised, triggering end of main loop.")
            raise
        finally:
            total_time = time.time()-start_time
            cpu_hr = total_time/60/60*SIZE
            truth.logger.info("Iterations: {:d}".format(truth.solver.iteration))
            truth.logger.info("Sim end time: {:.3e}".format(truth.solver.sim_time))
            truth.logger.info("Run time: {:.3e} sec".format(total_time))
            truth.logger.info("Run time: {:.3e} cpu-hr".format(cpu_hr))
            truth.logger.debug("END OF SIMULATION")

from RB_2D import *
from RB_2D_assim import *
from RB_2D_params import *
import numpy as np
from functools import partial
import json
import itertools
import os
import shutil
import sys
from tqdm import tqdm

#JPW notes: need to include definition and creation of self.dt_hist and self.self.prev_state (which will just be zeta)

#@staticmethod
class RB_2D_estimator(object):
    """A class to run data assimilations on the 2D RB system
    """

    def __init__(self, projector, outdir='new_run', L=4, xsize=384, zsize=192, Prandtl=1,
                 Rayleigh=1e6, mu = 1, N=32, warmup_time=2, warmup_dt=1e-5,
                 overwrite=False, final_sim_time=5):
        """Run normal 2D RB for warmup_time to get into the turbulent setting.
        """

        # The system we are gathering low-mode data from
        self.truth = RB_2D(L=L, xsize=xsize, zsize=zsize, Prandtl=Prandtl, Rayleigh=Rayleigh)
        self.truth.setup_simulation(wall_time=1e10, sim_time=warmup_time)

        # The assimalating system
        self.estimator = RB_2D_assim(projector=projector, mu=mu, N=N, L=L, xsize=xsize,
                                  zsize=zsize, Prandtl=Prandtl, Rayleigh=Rayleigh)
        self.estimator.setup_simulation(wall_time=1e10, sim_time=final_sim_time)
        #estimator.finalize_solver(truth.solver.state['zeta'])

        self.final_sim_time = final_sim_time

    def run_simulation(self):

        # Perform the warmup loop
        self.truth.logger.info("Starting warmup loop")
        self.truth.run_simulation()

        # Add the rest of the time to the truth solver
        self.truth.solver.stop_sim_time += self.final_sim_time

        # Run the simulation
        try:

            # Log entrance into main loop, start counter
            self.truth.logger.info("Starting main loop")
            self.estimator.logger.info("Starting main loop")
            start_time = time.time()

            # Iterate
            while self.truth.solver.ok & self.estimator.solver.ok:

                # Use CFL condition to compute time step
                dt = np.min([self.truth.cfl.compute_dt(), self.estimator.cfl.compute_dt()])

                # Step the truth simulation
                self.truth.solver.step(dt)

                # true state
                self.zeta = self.truth.solver.state['zeta']
                self.zeta.set_scales(1)

                # assimilating state
                self.zeta_assim = self.estimator.solver.state['zeta']
                self.zeta_assim.set_scales(1)

                # Get projection of difference between assimilating state and true state
                self.dzeta = self.estimator.problem.domain.new_field(name='dzeta')
                self.dzeta['g'] = self.zeta_assim['g'] - self.zeta['g']

                # Substitute this projection for the "driving" parameter in the assimilating system
                #self.estimator.problem.parameters["driving"].args = [self.dzeta, self.estimator.N]

                self.estimator.problem.parameters["driving"] = P_N(self.dzeta, self.estimator.N)

                print(self.estimator.problem.parameters["driving"])
                #estimator.problem.parameters["driving"].original_args = [dzeta, self.N]

                # Step the estimator
                self.estimator.solver.step(dt)

                # Record properties every tenth iteration
                if self.truth.solver.iteration % 10 == 0:

                    # Calculate max Re number
                    Re = self.truth.flow.max("Re")
                    Re_assim = self.estimator.flow.max("Re")

                    # Output diagnostic info to log
                    info = "Truth Iteration {:>5d}, Time: {:.7f}, dt: {:.2e}, Max Re = {:f}".format(
                        self.truth.solver.iteration, self.truth.solver.sim_time, dt, Re)
                    self.truth.logger.info(info)

                    # Output diagnostic info for assimilating system
                    info_assim = "Estimator iteration {:>5d}, Time: {:.7f}, dt: {:.2e}, Max Re = {:f}".format(
                        self.estimator.solver.iteration, self.estimator.solver.sim_time, dt, Re)
                    self.estimator.logger.info(info_assim)

                    if np.isnan(Re):
                        raise ValueError("Reynolds number went to infinity!!"
                                         "\nRe = {}", format(Re))
                    if np.isnan(Re_assim):
                        raise ValueError("Reynolds number went to infinity in assimilating system!!"
                                         "\nRe = {}", format(Re_assim))
        except BaseException as e:
            self.truth.logger.error("Exception raised, triggering end of main loop.")
            raise
        finally:
            total_time = time.time()-start_time
            cpu_hr = total_time/60/60*SIZE
            self.truth.logger.info("Iterations: {:d}".format(self.truth.solver.iteration))
            self.truth.logger.info("Sim end time: {:.3e}".format(self.truth.solver.sim_time))
            self.truth.logger.info("Run time: {:.3e} sec".format(total_time))
            self.truth.logger.info("Run time: {:.3e} cpu-hr".format(cpu_hr))
            self.truth.logger.debug("END OF SIMULATION")
#
#
#
#
#
#
#
#
#
#
#
#
#
#     def run_assimilator(self, outdir, final_sim_time=5):
#         estimator = self.estimator
#         truth = self.truth
#         estimator.solver.stop_sim_time = final_sim_time
#         estimator.solver.stop_wall_time = 1e10
# #        assimilator.solver.stop_iteration = np.inf
#         truth.solver.stop_sim_time += final_sim_time
#
#         try:
#             estimator.logger.info("Starting main loop")
#             start_time = time.time()
#             while truth.solver.ok & estimator.solver.ok:
#                 dt = np.min([truth.cfl.compute_dt(), estimator.cfl.compute_dt()])
#                 dt = truth.solver.step(dt)
#                 dzeta = estimator.problem.domain.new_field(name='dzeta')
#                 zeta = truth.solver.state['zeta']
#                 zeta.set_scales(1)
#                 zeta_assim = estimator.solver.state['zeta']
#                 zeta_assim.set_scales(1)
#                 dzeta['g'] = zeta_assim['g'] - zeta['g']
# #                dzeta['g'] = assimilator.solver.state['zeta']['g'] - truth.solver.state['zeta']['g']
#                 estimator.problem.parameters["driving"].args = [dzeta, estimator.N]
#                 #estimator.problem.parameters["driving"].original_args = [dzeta, self.N]
#
#
#                 dt = estimator.solver.step(dt)
#
#                 if truth.solver.iteration % 10 == 0:
#                     info = "Iteration {:>5d}, Time: {:.7f}, dt: {:.2e}".format(
#                         truth.solver.iteration, truth.solver.sim_time, dt)
#                     Re = truth.flow.max("Re")
#                     info += ", Max Re = {:f}".format(Re)
#                     truth.logger.info(info)
#                     info_assim = "Estimator iteration {:>5d}, Time: {:.7f}, dt: {:.2e}".format(
#                         estimator.solver.iteration, estimator.solver.sim_time, dt)
#                     Re_assim = estimator.flow.max("Re")
#                     info_assim += ", Max Re = {:f}".format(Re_assim)
#                     estimator.logger.info(info_assim)
#                     if np.isnan(Re):
#                         raise ValueError("Reynolds number went to infinity!!"
#                                          "\nRe = {}", format(Re))
#         except BaseException as e:
#             truth.logger.error("Exception raised, triggering end of main loop.")
#             raise
#         finally:
#             total_time = time.time()-start_time
#             cpu_hr = total_time/60/60*SIZE
#             truth.logger.info("Iterations: {:d}".format(truth.solver.iteration))
#             truth.logger.info("Sim end time: {:.3e}".format(truth.solver.sim_time))
#             truth.logger.info("Run time: {:.3e} sec".format(total_time))
#             truth.logger.info("Run time: {:.3e} cpu-hr".format(cpu_hr))
#             truth.logger.debug("END OF SIMULATION")

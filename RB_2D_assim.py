# RB_2D_assim.py
"""
Dedalus script for simulating the nudged 2D Rayleigh-Benard system.

Authors: , Jacob Murri, Jared Whitehead
"""

import os
import re
import h5py
import time
import numpy as np
from scipy.integrate import simps
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.animation import writers as mplwriters
try:
    from tqdm import tqdm
except ImportError:
    print("Recommended: install tqdm (pip install tqdm)")
    tqdm = lambda x: x

from dedalus import public as de
from dedalus.extras import flow_tools
from dedalus.core.operators import GeneralFunction

from base_simulator import BaseSimulator, RANK, SIZE
from RB_2D import RB_2D, P_N


class RB_2D_assim(RB_2D):
    """
    Manager for dedalus simulations of a nudged 2D Rayleigh-Benard system.

    Let Psi be defined on [0,L]x[0,1] with coordinates (x,z). Defining
    u = [v, w] = [-psi_z, psi_x] and zeta = laplace(psi),
    the nudging equations for assimilating a state to the Rayleigh-Benard system
    can be written

    Pr [Ra T_x + laplace(zeta)] - zeta_t = u.grad(zeta) - mu * driving
                        laplace(T) - T_t = u.grad(T)
    subject to
        u(z=0) = 0 = u(z=1)
        T(z=0) = 1, T(z=1) = 0
        u, T periodic in x (use a Fourier basis)

    Variables:
        u:R2xR -> R2: the fluid velocity vector field.
        T:R2xR -> R: the fluid temperature.
        p:R2xR -> R: the pressure.
        Ra: the Rayleigh number.
        Pr: the Prandtl number.
        mu: nudging constant.
        driving: typically taken to be P_N(zeta_estimated - zeta_true)
    """


    def setup_evolution(self, **kwargs):
        """
        Sets up the main Boussinesq evolution equations with nudging for the state, assuming
        all parameters, auxiliary equations, and boundary conditions are already
        defined.
        """
        ### Originally -mu*driving
        self.problem.add_equation("Pr*(Ra*dx(T) + dx(dx(zeta)) + dz(zetaz)) - dt(zeta) = v*dx(zeta) + w*zetaz - mu*driving")
        self.problem.add_equation("dt(T) - dx(dx(T)) - dz(Tz)= - v*dx(T) - w*Tz")

    def setup_params(self, L, xsize, zsize, Prandtl, Rayleigh, mu, N, **kwargs):
        """
        Sets up the parameters for the dedalus IVP problem. Does not set up the
        equations for the problem yet. Assumes self.problem is already defined.

        Parameters:
            L (float): the length of the x domain. In x and z, the domain is
                therefore [0,L]x[0,1].
            xsize (int): the number of points to discretize in the x direction.
            zsize (int): the number of points to discretize in the z direction.
            Prandtl (None or float): the ratio of momentum diffusivity to
                thermal diffusivity of the fluid. If None (default), then
                the system is set up as if Prandtl = infinity.
            Rayleigh (float): measures the amount of heat transfer due to
                convection, as opposed to conduction.
            mu (float): constant on the Fourier projection in the
                Data Assimilation system.
            N (int): the number of modes to keep in the Fourier projection.
        """
        # Domain parameters
        self.problem.parameters['L'] = L
        self.problem.parameters['xsize'] = xsize
        self.problem.parameters['zsize'] = zsize

        # Fluid parameters
        self.problem.parameters['Ra'] = Rayleigh
        self.problem.parameters['Pr'] = Prandtl

        # Driving parameters
        self.problem.parameters['mu'] = mu
        self.mu = mu
        self.problem.parameters['N'] = N
        self.N = N

        # GeneralFunction for driving
        self.problem.parameters["driving"] = GeneralFunction(self.problem.domain, 'g', P_N, args=[])

    def plot_convergence(self, savefig=True):
        """Plot the six measures of convergence over time."""
        # self.merge_results()
        datafile = self._get_merged_file("analysis")
        self.logger.info("Plotting convergence estimates from '{}'...".format(
                                                                    datafile))
        # Gather data from the source file.
        with h5py.File(datafile, 'r') as data:
            times = list(data["scales/sim_time"])
            T_err = data["tasks/T_err"][:,0,0]
            gradT_err = data["tasks/gradT_err"][:,0,0]
            u_err = data["tasks/u_err"][:,0,0]
            gradu_err = data["tasks/gradu_err"][:,0,0]
            T_h2_err = data["tasks/T_h2_err"][:,0,0]
            u_h2_err = data["tasks/u_h2_err"][:,0,0]
            T_L2 = data["tasks/T_L2"][:,0,0]
            gradT_L2 = data["tasks/gradT_L2"][:,0,0]
            u_L2 = data["tasks/u_L2"][:,0,0]
            gradu_L2 = data["tasks/gradu_L2"][:,0,0]
            T_h2 = data["tasks/T_h2"][:,0,0]
            u_h2 = data["tasks/u_h2"][:,0,0]

        with plt.style.context(".mplstyle"):
            # Make subplots and a big plot for an overlay.
            fig = plt.figure(figsize=(12,6))
            ax1 = plt.subplot2grid((3,4), (0,0))
            ax2 = plt.subplot2grid((3,4), (0,1))
            ax3 = plt.subplot2grid((3,4), (1,0))
            ax4 = plt.subplot2grid((3,4), (1,1))
            ax5 = plt.subplot2grid((3,4), (2,0))
            ax6 = plt.subplot2grid((3,4), (2,1))
            axbig = plt.subplot2grid((3,4), (0,2), rowspan=3, colspan=2)

            # Plot the data.
            ax1.semilogy(times, T_err/T_L2[0], 'C0', lw=.5)
            ax2.semilogy(times, u_err/u_L2[0], 'C1', lw=.5)
            ax3.semilogy(times, gradT_err/gradT_L2[0], 'C2', lw=.5)
            ax4.semilogy(times, gradu_err/gradu_L2[0], 'C3', lw=.5)
            ax5.semilogy(times, T_h2_err/T_h2[0], 'C4', lw=.5)
            ax6.semilogy(times, u_h2_err/u_h2[0], 'C5', lw=.5)
            axbig.semilogy(times, T_err/T_L2[0], 'C0', lw=.5,
                           label=r"$||(\tilde{T} - T)(t)||_{L^2(\Omega)}$")
            axbig.semilogy(times, u_err/u_L2[0], 'C1', lw=.5,
                           label=r"$||(\tilde{\mathbf{u}} - \mathbf{u})(t)||"
                                 r"_{L^2(\Omega)}$")
            axbig.semilogy(times, gradT_err/gradT_L2[0], 'C2', lw=.5,
                           label=r"$||(\nabla\tilde{T} - \nabla T)(t)||"
                                 r"_{L^2(\Omega)}$")
            axbig.semilogy(times, gradu_err/gradu_L2[0], 'C3', lw=.5,
                           label=r"$||(\nabla\tilde{\mathbf{u}} - \nabla"
                                 r"\mathbf{u})(t)||_{L^2(\Omega)}$")
            axbig.semilogy(times, T_h2_err/T_h2[0], 'C4', lw=.5,
                           label=r"$||(\tilde{T} - T)(t)||_{H^2(\Omega)}$")
            axbig.semilogy(times, u_h2_err/u_h2[0], 'C5', lw=.5,
                           label=r"$||(\tilde{\mathbf{u}} - \mathbf{u})(t)||"
                                 r"_{H^2(\Omega)}$")
            axbig.legend(loc="upper right")

            # Set minimal axis and tick labels.
            for ax in [ax1, ax2, ax3, ax4]:
                ax.set_xticklabels([])
            for ax in [ax2, ax4, ax6]:
                ax.set_yticklabels([])
            ax5.set_xlabel("Simulation Time", color="white")
            ax6.set_xlabel("Simulation Time", color="white")
            axbig.set_xlabel("Simulation Time", color="white")
            fig.text(0.5, 0.01, r"Simulation Time $t$", ha="center",
                     fontsize=16)
            ax1.set_title(r"$||(\tilde{T} - T)(t)||_{L^2(\Omega)}$")
            ax2.set_title(r"$||(\tilde{\mathbf{u}} - \mathbf{u})(t)||"
                          r"_{L^2(\Omega)}$")
            ax3.set_title(r"$||(\nabla\tilde{T} - \nabla T)(t)||"
                          r"_{L^2(\Omega)}$")
            ax4.set_title(r"$||(\nabla\tilde{\mathbf{u}} - \nabla"
                          r"\mathbf{u})(t)||_{L^2(\Omega)}$")
            ax5.set_title(r"$||(\tilde{T} - T)(t)||_{H^2(\Omega)}$")
            ax6.set_title(r"$||(\tilde{\mathbf{u}} - \mathbf{u})(t)||"
                          r"_{H^2(\Omega)}$")
            axbig.set_title("Overlay")

            # Make the axes uniform and use tight spacing.
            xlims = axbig.get_xlim()
            for ax in [ax1, ax2, ax3, ax4, ax5, ax6, axbig]:
                ax.set_xlim(xlims)
                ax.set_ylim(1e-13, 1e1)
            plt.tight_layout()

            # Save or show the figure.
            if savefig:
                outfile = os.path.join(self.records_dir, "convergence.png")
                plt.savefig(outfile, dpi=300, bbox_inches="tight")
                self.logger.info("\tFigure saved as '{}'".format(outfile))
            else:
                plt.show()
            plt.close()

    def plot_nusselt(self, savefig=True):
        """Plot the three measures of the Nusselt number over time for the
        base and DA systems.
        """
        # self.merge_results()
        datafile = self._get_merged_file("analysis")
        self.logger.info("Plotting Nusselt number from '{}'...".format(
                                                                    datafile))
        # Gather data from the source file.
        times = []
        nusselt = [[] for _ in range(6)]
        with h5py.File(datafile, 'r') as data:
            times = list(data["scales/sim_time"])
            for i in range(1,4):
                label = "tasks/Nu_{}".format(i)
                nusselt[i-1] = data[label][:,0,0]
                nusselt[i+2] = data[label+"_da"][:,0,0]
        t, nusselt = np.array(times), np.array(nusselt)

        # Calculate time averages (integrate using Simpson's rule).
        nuss_avg = np.array([[simps(nu[:n], t[:n]) for n in range(1,len(t)+1)]
                                                            for nu in nusselt])
        nuss_avg[:,1:] /= t[1:]

        with plt.style.context(".mplstyle"):
            # Plot results in 4 subplots (raw nusselt vs time avg, nonDA vs DA)
            fig = plt.figure(figsize=(12,6))
            ax1 = plt.subplot2grid((2,4), (0,0))
            ax2 = plt.subplot2grid((2,4), (0,1), sharey=ax1)
            ax3 = plt.subplot2grid((2,4), (1,0))
            ax4 = plt.subplot2grid((2,4), (1,1), sharey=ax3)
            axbig = plt.subplot2grid((2,4), (0,2), rowspan=2, colspan=2)
            for i in [0,1,2]:
                ax1.plot(t[1:], nusselt[i,1:])
                ax3.plot(t[1:], nuss_avg[i,1:])
                ax2.plot(t[1:], nusselt[i+3,1:])
                ax4.plot(t[1:], nuss_avg[i+3,1:])
            axbig.plot(t[1:], nuss_avg[:3,1:].mean(axis=0),
                       label='Data ("Truth")')
            axbig.plot(t[1:], nuss_avg[3:,1:].mean(axis=0),
                       label="Assimilating System")
            ax1.set_title("Raw Nusselt", fontsize=8)
            ax3.set_title("Time Average", fontsize=8)
            ax2.set_title("DA Raw Nusselt", fontsize=8)
            ax4.set_title("DA Time Average", fontsize=8)
            axbig.set_title("Overlay of Mean Time Averages", fontsize=8)
            axbig.legend(loc="lower right")
            plt.tight_layout()

            if savefig:
                outfile = os.path.join(self.records_dir, "nusselt.png")
                plt.savefig(outfile, dpi=300, bbox_inches="tight")
                self.logger.info("\tFigure saved as '{}'".format(outfile))
            else:
                plt.show()
            plt.close()

    def animate_temperature(self, max_frames=np.inf, fps=100):
        """Animate the temperature results of the simulation (model and DA
        system) and save it to an mp4 file called 'temperature.mp4'.
        """
        # self.merge_results()
        state_file = self._get_merged_file("states")
        self.logger.info("Creating temperature animation from '{}'...".format(
                                                                state_file))

        # Set up the figure / movie writer.
        fig = plt.figure(figsize=(12,6))
        ax1 = plt.subplot2grid((2,2), (0,0))
        ax2 = plt.subplot2grid((2,2), (0,1))
        ax4 = plt.subplot2grid((2,2), (1,0), colspan=2)
        # fig, [[ax1, ax3], [ax2, ax4]] = plt.subplots(2, 2)
        ax1.axis("off"); ax2.axis("off") #; ax3.axis("off")
        ax1.set_title('Data ("Truth")')
        ax2.set_title("Assimilating System")
        # ax3.set_title("Projected Temperature Difference", fontsize=8)
        writer = mplwriters["ffmpeg"](fps=fps) # frames per second, sets speed.

        # Rename the old animation if it exists (it will be deleted later).
        outfile = os.path.join(self.records_dir, "temperature.mp4")
        oldfile = os.path.join(self.records_dir, "old_temperature.mp4")
        if os.path.isfile(outfile):
            self.logger.info("\tRenaming old animation '{}' -> '{}'".format(
                                                            outfile, oldfile))
            os.rename(outfile, oldfile)

        # Write the movie at 200 DPI (resolution).
        with writer.saving(fig,outfile,200), h5py.File(state_file,'r') as data:
            print("Extracting data...", end='', flush=True)
            T = data["tasks/T"]
            T_ = data["tasks/T_"]
            # dT = data["tasks/P_N"]
            times = list(data["scales/sim_time"])
            assert len(times) == len(T) == len(T_), "mismatched dimensions"
            print("done")

            # Plot ||T_ - T||_L^infinity.
            print("Calculating / plotting ||T_ - T||_L^infty(Omega)...",
                  end='', flush=True)
            L_inf = np.max(np.abs(T_[:] - T[:]), axis=(1,2))
            ax4.semilogy(times, L_inf, lw=1)
            ax4_line = plt.axvline(x=times[0], color='r', lw=.5)
            _, ylims = ax4_line.get_data()
            ax4.set_xlim(times[0], times[-1])
            ax4.set_ylim(1e-11, 1e1)
            ax4.set_title(r"$||\tilde{T} - T||_{L^\infty(\Omega)} =$" \
                          + "{:.2e}".format(L_inf[0]))
            ax4.spines["right"].set_visible(False)
            ax4.spines["top"].set_visible(False)
            ax4.set_xlabel(r"Simulation Time $t$")
            print("done")

            # Set up color maps for each temperature layer.
            im1 = ax1.imshow( T[0].T, animated=True, cmap="inferno",
                             vmin=0, vmax=1)
            im2 = ax2.imshow(T_[0].T, animated=True, cmap="inferno",
                             vmin=0, vmax=1)
            # im3 = ax3.imshow(dT[0].T, animated=True, cmap="RdBu_r",
            #                  vmin=-.05, vmax=.05)
                             # norm=SymLogNorm(linthresh=1e-10, vmin=-1, vmax=1))
            # im3 = ax3.imshow(np.log(np.abs(T[0] - T_[0]) + 1e-16).T,
            #                  animated=True, cmap="viridis") # log difference
            # fig.colorbar(im3, ax=ax3, fraction=0.023)
            ax1.invert_yaxis() # Flip the images right-side up.
            ax2.invert_yaxis()
            # ax3.invert_yaxis()

            # Save a frame for each layer of task data.
            for j in tqdm(range(min(T.shape[0], max_frames))):
                im1.set_array( T[j].T)     # Truth
                im2.set_array(T_[j].T)     # Approximation
                # im3.set_array(dT[j].T)     # Difference
                # im3.set_array(np.log(np.abs(T[j] - T_[j]) + 1e-16).T)

                # Moving line for ||T - T_||_L^infty error plot.
                t = times[j]
                ax4_line.set_data([[t,t], ylims])
                ax4.set_title(r"$||(\tilde{T}-T)(t)||_{L^\infty(\Omega)} =$" \
                              + "{:.2e}".format(L_inf[j]))
                writer.grab_frame()
        self.logger.info("\tAnimation saved as '{}'".format(outfile))
        plt.close()

        # Delete the old animation.
        if os.path.isfile(oldfile):
            self.logger.info("\tDeleting old animation '{}'".format(oldfile))
            os.remove(oldfile)


# RB_2D_assimilator ============================================================

class RB_2D_assimilator(object):
    """
    A class to run data assimilation on the 2D RB system.
    """

    def __init__(self, L=4, xsize=384, zsize=192, Prandtl=1, Rayleigh=1e6, mu = 1, N=32, BCs = 'no-slip', **kwargs):

        """
        Creates an instance of RB_2D representing the "true" system which is being
        measured, and an instance of RB_2D_assim representing the nudged system
        which converges to the state of the true system.

        Required Parameters:
            L (float): the length of the x domain. In x and z, the domain is
                therefore [0,L]x[0,1].
            xsize (int): the number of points to discretize in the x direction.
            zsize (int): the number of points to discretize in the z direction.
            Prandtl (None or float): the ratio of momentum diffusivity to
                thermal diffusivity of the fluid. If None (default), then
                the system is set up as if Prandtl = infinity.
            Rayleigh (float): measures the amount of heat transfer due to
                convection, as opposed to conduction.
            mu (float): constant on the Fourier projection in the
                Data Assimilation system.
            N (int): the number of modes to keep in the Fourier projection.
            BCs (str): if 'no-slip', use the no-slip BCs u(z=0,1) = 0.
                If 'free-slip', use the free-slip BCs u_z(z=0,1) = 0.
        """

        # The system we are gathering low-mode data from
        self.truth = RB_2D(L=L, xsize=xsize, zsize=zsize, Prandtl=Prandtl, Rayleigh=Rayleigh, BCs=BCs, **kwargs)

        # The assimalating system
        self.estimator = RB_2D_assim(mu=mu, N=N, L=L, xsize=xsize, zsize=zsize, Prandtl=Prandtl, Rayleigh=Rayleigh, BCs=BCs, **kwargs)

    def setup_simulation(self, scheme=de.timesteppers.RK443, warmup_time=2, final_sim_time=5, wall_time=1e10, stop_iteration=np.inf,
                        tight=False, save=.05, save_tasks=None, analysis=True, analysis_tasks=None, initial_conditions1=None, initial_conditions2=None, **kwargs):
        """
        scheme (string, de.timestepper): The kind of solver to use. Options are
            RK443 (de.timesteppers.RK443), RK111, RK222, RKSMR, etc.
        warmup_time (float): The maximum amount of warmup time allowed
            (in seconds) before ending the simulation.
        final_sim_time (float): The maximum amount of simulation time allowed
            (in seconds) before ending the simulation.
        wall_time (float): The maximum amound of computing time allowed
            (in seconds) before ending the simulation.
        stop_iteration (numeric): The maximum amount of iterations allowed
            before ending the simulation
        save (float): The number of simulation seconds that pass between
            saving the state files. Higher save result in smaller data
            files, but lower numbers result in better animations.
            Set to 0 to disable saving state files.
        save_tasks (list of str): which state variables to save. If None,
            uses ['T', 'Tz', 'psi', 'psiz', 'zeta', 'zetaz'].

        analysis (bool): Whether or not to track convergence measurements.
            Disable for faster simulations (less message passing via MPI)
            when convergence estimates are not needed (i.e. movie only).
        analysis_tasks (list of 2-tuples of strs): which analysis tasks to
            perform, given as a list of (task, name) pairs. If None, uses a
            default list.

        initial_conditions (None, str): determines from what source to
            draw the initial conditions. Valid options are as follows:
            - None: use trivial conditions (T_ = 1 - z, T = 1 - z + eps).
            - 'resume': use the most recent state file in the
                records directory (load both model and DA system).
            - An .h5 filename: load state variables for the model and
                reset the data assimilation state variables to zero.

        #### Not Implemented
        tight (bool): If True, set a low cadence and min_dt for refined
            simulation. If False, set a higher cadence and min_dt for a
            more coarse (but faster) simulation.
        """
        # The system we are gathering low-mode data from
        self.truth.setup_simulation(scheme=scheme, sim_time=warmup_time, wall_time=wall_time, stop_iteration=stop_iteration, tight=tight,
                           save=save, save_tasks=save_tasks, analysis=analysis, analysis_tasks=analysis_tasks, initial_conditions=initial_conditions1, **kwargs)

        # The assimalating system
        self.estimator.setup_simulation(scheme=scheme, sim_time=final_sim_time, wall_time=wall_time, stop_iteration=stop_iteration, tight=tight,
                           save=save, save_tasks=save_tasks, analysis=analysis, analysis_tasks=analysis_tasks, initial_conditions=initial_conditions2, **kwargs)

        # Record final_sim_time
        self.final_sim_time = final_sim_time

    def run_simulation(self):
        """
        Runs the simulations and performs data assimilation. First puts the "true"
        system through a warmup loop, then steps through the true and estimator
        systems simultaneously, updating parameters in the estimating system at
        each time step.
        """

        # Perform the warmup loop
        self.truth.logger.info("Starting warmup loop")
        self.truth.run_simulation()

        # Add the rest of the time to the truth solver
        self.truth.solver.stop_sim_time += self.final_sim_time

        # Set initial conditions for estimating system to simply be projections of the initial state of the estimating system
        # for variable in self.truth.problem.variables:

        #     # Set scales appropriately
        #     self.estimator.solver.state[variable].set_scales(3/2)
        #     self.truth.solver.state[variable].set_scales(3/2)

        #     self.estimator.solver.state[variable]['g'] = P_N(self.truth.solver.state[variable], self.estimator.N)

        # Run the simulation
        try:

            # Log entrance into main loop, start counter
            self.truth.logger.info("Starting main loop")
            self.estimator.logger.info("Starting main loop")
            start_time = time.time()

            # Iterate
            while self.truth.solver.ok & self.estimator.solver.ok:

                # Use CFL condition to compute time step
                self.dt = np.min([self.truth.cfl.compute_dt(), self.estimator.cfl.compute_dt()])

                # Step the truth simulation
                self.truth.solver.step(self.dt)

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
                if self.estimator.solver.iteration == 0: self.estimator.problem.parameters["driving"].original_args = [self.dzeta, self.estimator.N]
                self.estimator.problem.parameters["driving"].args = [self.dzeta, self.estimator.N]

                # Step the estimator

                self.estimator.solver.step(self.dt)

                # Record properties every tenth iteration
                if self.truth.solver.iteration % 10 == 0:

                    # Calculate max Re number
                    Re = self.truth.flow.max("Re")
                    Re_assim = self.estimator.flow.max("Re")

                    # Output diagnostic info to log
                    info = "Truth Iteration {:>5d}, Time: {:.7f}, dt: {:.2e}, Max Re = {:f}".format(
                        self.truth.solver.iteration, self.truth.solver.sim_time, self.dt, Re)
                    self.truth.logger.info(info)

                    # Output diagnostic info for assimilating system
                    info_assim = "Estimator iteration {:>5d}, Time: {:.7f}, dt: {:.2e}, Max Re = {:f}".format(
                        self.estimator.solver.iteration, self.estimator.solver.sim_time, self.dt, Re)
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

            self.truth.merge_results()
            self.estimator.merge_results()

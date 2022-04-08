# RB_2D_params.py
"""
Dedalus script for simulating the nudged and parameter updated 2D RB system.

Authors: Jacob Murri, Jared Whitehead
"""

import os
import re
import h5py
import time
import numpy as np
from scipy.integrate import simps
try:
    from tqdm import tqdm
except ImportError:
    print("Recommended: inastall tqdm (pip install tqdm):")
    tqdm = lambda x: x

from dedalus import public as de
from dedalus.extras import flow_tools
from dedalus.core.operators import GeneralFunction

from base_simulator import BaseSimulator, RANK, SIZE
from RB_2D import RB_2D, P_N
from RB_2D_assim import RB_2D_assim, RB_2D_assimilator



class RB_2D_params(RB_2D_assim):
    """
    Manager for dedalus simulations of a nudged, parameter estimating 2D Rayleigh-Benard system.

    Let Psi be defined on [0,L]x[0,1] with coordinates (x,z). Defining
    u = [v, w] = [-psi_z, psi_x] and zeta = laplace(psi),
    the nudging equations for assimilating a state to the Rayleigh-Benard system
    can be written

    Pr~ [Ra~ T_x + laplace(zeta)] - zeta_t = u.grad(zeta) - mu * driving
                        laplace(T) - T_t = u.grad(T)
    subject to
        u(z=0) = 0 = u(z=1)
        T(z=0) = 1, T(z=1) = 0
        u, T periodic in x (use a Fourier basis)

    Variables:
        u:R2xR -> R2: the fluid velocity vector field.
        T:R2xR -> R: the fluid temperature.
        p:R2xR -> R: the pressure.
        Ra~: the (estimated) Rayleigh number.
        Pr~: the (estimated)Prandtl number.
        mu: nudging constant.
        driving: typically taken to be P_N(zeta_estimated - zeta_true)
    """


    def setup_evolution(self, **kwargs):
        """
        Sets up the main Boussinesq evolution equations with nudging for the state
        and parameter estimation,  assuming all parameters, auxiliary equations,
        and boundary conditions are already defined.
        """

        self.problem.add_equation("Pr*(Ra*dx(T) + dx(dx(zeta)) + dz(zetaz)) - dt(zeta) = v*dx(zeta) + w*zetaz - mu*driving - PrRa_coeff*dx(T) - Pr_coeff*(dx(dx(zeta)) + dz(zetaz))")
        self.problem.add_equation("dt(T) - dx(dx(T)) - dz(Tz) = -v*dx(T) - w*Tz")

    def const_val(self, value, return_field=False):
        """
        Assuming that the problem is already defined, create a new field on the
        problem's domain with constant value given by the argument value.

        Parameters:
            value (numeric): the value which will be given to the new field.
        """
        coefficient_field = self.problem.domain.new_field()
        coefficient_field['g'] = value
        if return_field:
            return coefficient_field #somehow need to make this so that the field isn't defined every time it is called, and the domain is available...maybe define a field at the initial step and then have that stored in self, then adjusted as necessary?
        else:
            coefficient_field.set_scales(3/2)
            return coefficient_field['g']

    def setup_params(self, L, xsize, zsize, Prandtl, Rayleigh, mu, N, Pr_guess, Ra_guess, alpha=1, **kwargs):
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
            alpha (float): Relaxation coefficient
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
        self.problem.parameters['driving'] = GeneralFunction(self.problem.domain, 'g', P_N, args=[])

        # Parameter estimation
        self.problem.parameters['Pr_coeff'] = GeneralFunction(self.problem.domain, 'g', self.const_val, args=[])
        self.problem.parameters['PrRa_coeff'] = GeneralFunction(self.problem.domain, 'g', self.const_val, args=[])
        self.Pr_guess = Pr_guess
        self.Ra_guess = Ra_guess

        # Store a little history for estimation of backwards time derivative
        #self.problem.parameters['Pr_coeff'].args = [0.]
        #self.problem.parameters['Pr_coeff'].original_args = [0.]
        #self.problem.parameters['PrRa_coeff'].args = [0.]
        #self.problem.parameters['PrRa_coeff'].orginal_args = [0.]

        #Need to initialize the previous 2 states and corresponding time steps
        self.prev_state = [None] * 2
        self.dt_hist = [None] * 2

        # Relaxation coefficient
        self.alpha = alpha

    def setup_simulation(self, scheme=de.timesteppers.RK443, sim_time=0.15, wall_time=60, stop_iteration=np.inf, tight=False,
                       save=.05, save_tasks=None, analysis=True, analysis_tasks=None, initial_conditions=None, **kwargs):
        """
        Load initial conditions, run the simulation, and merge results.

        Parameters:
            scheme (string, de.timestepper): The kind of solver to use. Options are
                RK443 (de.timesteppers.RK443), RK111, RK222, RKSMR, etc.
            sim_time (float): The maximum amount of simulation time allowed
                (in seconds) before ending the simulation.
            wall_time (float): The maximum amound of computing time allowed
                (in seconds) before ending the simulation.
            stop_iteration (numeric): The maximum amount of iterations allowed
                before ending the simulation

            #### Not Implemented
            tight (bool): If True, set a low cadence and min_dt for refined
                simulation. If False, set a higher cadence and min_dt for a
                more coarse (but faster) simulation.

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
        """

        # Log a new simulation
        self.logger.debug("\n")
        self.logger.debug("NEW SIMULATION")

        # Interpret the scheme, if necessary
        schemes = {'RK443': de.timesteppers.RK443, 'MCNAB2': de.timesteppers.MCNAB2, 'SBDF3': de.timesteppers.SBDF3}
        if type(scheme) == str: scheme = schemes[scheme]

        # Build the solver
        self.solver = self.problem.build_solver(scheme)

        # Set up initial conditions
        self.setup_ic(initial_conditions)

        # State snapshots -----------------------------------------------------
        if save:

            # Save the temperature measurements in states/ files. Use sim_dt.
            self.snaps = self.solver.evaluator.add_file_handler(
                                    os.path.join(self.records_dir, "states"),
                                    sim_dt=save, max_writes=5000, mode="append")
                                    # Set save=0.005 or lower for more writes.

            # Default save tasks
            if save_tasks is None:
                save_tasks = ['T', 'Tz', 'psi', 'psiz', 'zeta', 'zetaz', 'Pr_coeff', 'PrRa_coeff']

            for task in save_tasks: self.snaps.add_task(task)

        # Convergence analysis ------------------------------------------------
        if analysis:
            # Save specific tasks in analysis/ files every few iterations.
            self.annals = self.solver.evaluator.add_file_handler(
                                    os.path.join(self.records_dir, "analysis"),
                                    iter=20, max_writes=73600, mode="append")

            # Default analysis tasks
            if analysis_tasks is None:
                analysis_tasks = [
                                  ("1 + integ(w*T , 'x', 'z')/L", "Nu_1"),
                                  ("integ(dx(T)**2 + Tz**2, 'x', 'z')/L", "Nu_2"),
                                  ("integ(dx(v)**2 + dz(v)**2 + dx(w)**2 + dz(w)**2, 'x', 'z')", "Nu_3"),
                                  ("sqrt(integ(T**2, 'x', 'z'))", "T_L2"),
                                  ("sqrt( integ(dx(T)**2 + dz(T)**2, 'x', 'z'))", "gradT_L2"),
                                  ("sqrt( integ(v**2 + w**2, 'x', 'z'))", "u_L2"),
                                  ("sqrt( integ(dx(v)**2 + dz(v)**2 + dx(w)**2 + dz(w)**2, 'x', 'z'))", "gradu_L2"),
                                  ("sqrt( integ(dx(dx(T))**2 + dx(dz(T))**2 + dz(dz(T))**2, 'x', 'z'))", "T_h2"),
                                  ("sqrt(integ( dx(dx(v))**2 + dz(dz(v))**2 + dx(dz(v))**2 + dx(dz(w))**2 + dx(dx(w))**2 + dz(dz(w))**2, 'x','z'))", "u_h2")
                                 ]

            for task, name in analysis_tasks: self.annals.add_task(task, name=name)

        # Control Flow --------------------------------------------------------
        if scheme == de.timesteppers.MCNAB2:
            self.cfl = flow_tools.CFL(self.solver, initial_dt=self.dt, cadence=5, safety=.2,
                                 max_change=1.4, min_change=0.2,
                                 max_dt=0.01, min_dt=1e-11)
        else:
            self.cfl = flow_tools.CFL(self.solver, initial_dt=self.dt, cadence=10, safety=.5,
                                 max_change=1.5, min_change=0.5,
                                 max_dt=0.01,    min_dt=1e-8)

        self.cfl.add_velocities(('v',  'w' ))

        # Flow properties (print during run; not recorded in the records files)
        self.flow = flow_tools.GlobalFlowProperty(self.solver, cadence=1)
        self.flow.add_property("sqrt(v **2 + w **2) / Ra", name='Re' )

        # Set solver attributes
        self.solver.stop_sim_time = sim_time
        self.solver.stop_wall_time = wall_time
        self.solver.stop_iteration = stop_iteration

        # Set a flag
        self.solver_setup = True

    def new_params(self, zeta):
        """
        Method to estimate the parameters at each step, devised by B. Pachev and
        J. Whitehead.

        Parameters:
            zeta (dedalus field): The true system state
        """

        zeta.set_scales(3/2)

        # Get estimated state and backward time derivative
        zeta_tilde = self.solver.state['zeta']
        zeta_tilde.set_scales(3/2)
        zeta_t = self.backward_time_derivative()
        zeta_t.set_scales(3/2)

        # set up the alpha_i coefficients
        Ih_laplace_zeta = self.problem.domain.new_field()
        Ih_laplace_zeta.set_scales(3/2)
        laplace_zeta = (zeta_tilde.differentiate(x=2)+zeta_tilde.differentiate(z=2))
        Ih_laplace_zeta['g'] = P_N(laplace_zeta, self.N)

        # set up the beta_i coefficients
        Ih_temp_x = self.problem.domain.new_field()
        Ih_temp_x.set_scales(3/2)
        Ih_temp_x['g'] = P_N(self.solver.state['T'].differentiate(x=1),self.N)

        # set up the gamma_i coefficients
        remainder = self.problem.domain.new_field()
        remainder.set_scales(3/2)
        nonlinear_term = self.problem.domain.new_field()
        nonlinear_term.set_scales(3/2)
        zeta_error = self.problem.domain.new_field()
        zeta_error.set_scales(3/2)
        Ih_remainder = self.problem.domain.new_field()
        Ih_remainder.set_scales(3/2)

        zeta_error['g'] = zeta['g'] - zeta_tilde['g']

        # v = -psi_z, w = psi_x
        nonlinear_term['g'] = -self.solver.state['psi'].differentiate(z=1)['g']*zeta_tilde.differentiate(x=1)['g'] + self.solver.state['psi'].differentiate(x=1)['g']*zeta_tilde.differentiate(z=1)['g']
        remainder['g'] = nonlinear_term['g'] + zeta_t['g']
        Ih_remainder['g'] = P_N(remainder, self.N)

        e1 = self.problem.domain.new_field()
        e1.set_scales(3/2)
        e2 = self.problem.domain.new_field()
        e2.set_scales(3/2)

        #set e1 to be the projection of the error, guaranteeing exponential decay of the error
        e1['g'] = P_N(zeta_error, self.N)#JPW: check the order of this, should it be zeta_tilde-zeta?
        e2['g'] = Ih_temp_x['g']#Ih_laplace_zeta['g'] #start with this choice...others are possible
        #JPW: use I_h(error) as e_1, then start with something already computed, i.e. laplace operator (probably not nonlinear term...stick with linear differential operator ie don't square anything).  Use modified Gram-Schmidt to force this to be orthogonal to e_1 to give the 2nd direction

        # now perform modified Gram-Schmidt on e1 and e2 (no need to normalize I don't think)
        c = de.operators.integrate(e1*e2,'x','z')['g']*e1['g']
        e2['g'] = e2['g'] - c

        # Now e1 and e2 should be orthogonal. Calculate the coefficients
        alpha1 = de.operators.integrate(e1*Ih_laplace_zeta, 'x', 'z')['g'][0,0]
        alpha2 = de.operators.integrate(e2*Ih_laplace_zeta, 'x', 'z')['g'][0,0]

        beta1 = de.operators.integrate(e1*Ih_temp_x, 'x', 'z')['g'][0,0]
        beta2 = de.operators.integrate(e2*Ih_temp_x, 'x', 'z')['g'][0,0]

        gamma1 = -self.mu * de.operators.integrate(e1**2, 'x', 'z')['g'][0,0] + de.operators.integrate(e1*Ih_remainder, 'x', 'z')['g'][0,0]
        gamma2 = de.operators.integrate(e2*Ih_remainder, 'x', 'z')['g'][0,0]

        # Set up linear system and solve
        A = np.array([[alpha1, beta1], [alpha2, beta2]])
        b = np.array([[gamma1], [gamma2]])
        #print()
        #print('Matrix: ', A, 'Vector: ', b)
        #print()

        Pr, PrRa = np.linalg.solve(A,b)
        return float(Pr), float(PrRa/Pr)

        # try:
        #     Pr, PrRa = np.linalg.solve(A,b)
        #     # Return estimated coefficients
        #     return float(Pr), float(PrRa/Pr)
        # except: # if the matrix is singular
        #     print('Matrix is singular')
        #     return 0., 0.

    def backward_time_derivative(self):
        """
        Calculuate the time derivative of the observed vorticity based off of
        the estimating system. Derivation of the derivative approximation is made
        in the main document.
        """

        ### NOTE: Use projection of true state.

        if None in self.dt_hist:
            # There may be a better way to do this.  Right now zeta_t=0 for the first 2 time steps
            zeta_t = self.const_val(0., return_field=True)

        else:
            dt = self.dt_hist
            dt_ratio = dt[-1]/dt[-2]
            dt_diff = dt[-1]-dt[-2]
            dt_sum = dt[-1]+dt[-2]

            # Now to set up the coefficients of the finite differencing
            c0 = - (dt_diff*(dt[-1]**2 + dt_sum**2))/(dt[-1]*dt[-2])
            c1 = (dt_diff*dt_sum**2)/(dt[-1]*dt[-2])
            c2 = dt_ratio*dt_diff

            #Now using these coefficients we compute a 2nd order backward difference
            #approximation to the derivative of the vorticity
            zeta_t = self.problem.domain.new_field()

            # Make sure scales are set
            zeta_t.set_scales(3/2)
            self.solver.state['zeta'].set_scales(3/2)
            self.prev_state[-1].set_scales(3/2)
            self.prev_state[-2].set_scales(3/2)

            # Calculate
            zeta_t['g'] = c0*self.solver.state['zeta']['g'] + c1*self.prev_state[-1]['g'] + c2*self.prev_state[-2]['g']

            zeta_t.set_scales(3/2)

        return zeta_t

# RB_2D_estimator ==============================================================

class RB_2D_estimator(RB_2D_assimilator):
    """
    A class to run data assimilation and parameter recovery on the 2D RB system.
    """

    def __init__(self, L=4, xsize=384, zsize=192, Prandtl=1, Rayleigh=1e6, mu =1e3, N=32, BCs = 'no-slip', **kwargs):
        """
        Creates an instance of RB_2D representing the "true" system which is being
        measured, and an instance of RB_2D_params representing the nudged system
        which converges to the state and true parameters of the true system.

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
        self.truth = RB_2D(L=L, xsize=xsize, zsize=zsize, Prandtl=Prandtl, Rayleigh=Rayleigh, **kwargs)

        # The assimalating system
        self.estimator = RB_2D_params(mu=mu, N=N, L=L, xsize=xsize, zsize=zsize, Prandtl=Prandtl, Rayleigh=Rayleigh, **kwargs)

    
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
        for variable in self.truth.problem.variables:

            # Set scales appropriately
            self.estimator.solver.state[variable].set_scales(3/2)
            self.truth.solver.state[variable].set_scales(3/2)

            self.estimator.solver.state[variable]['g'] = P_N(self.truth.solver.state[variable], self.estimator.N)

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
                self.zeta.set_scales(3/2)

                # assimilating state
                self.zeta_assim = self.estimator.solver.state['zeta']
                self.zeta_assim.set_scales(3/2)

                # Get projection of difference between assimilating state and true state
                self.dzeta = self.estimator.problem.domain.new_field(name='dzeta')
                self.dzeta.set_scales(3/2)
                self.dzeta['g'] = self.zeta_assim['g'] - self.zeta['g']

                # Substitute this projection for the "driving" parameter in the assimilating system
                if self.estimator.solver.iteration == 0:
                    self.estimator.problem.parameters["driving"].original_args = [self.dzeta, self.estimator.N]
                self.estimator.problem.parameters["driving"].args = [self.dzeta, self.estimator.N]

                # Update the Parameters
                new_Pr_est, new_Ra_est = self.estimator.new_params(self.zeta)

                if self.estimator.solver.iteration == 0:
                    
                    # Use inital guess
                    Pr_coeff_initial = self.estimator.Pr_guess - self.truth.problem.parameters['Pr']
                    PrRa_coeff_initial = self.estimator.Pr_guess * self.estimator.Ra_guess - self.truth.problem.parameters['Pr']*self.truth.problem.parameters['Ra']

                    # Set parameters for the first time
                    self.estimator.problem.parameters['Pr_coeff'].original_args = [Pr_coeff_initial]
                    self.estimator.problem.parameters['PrRa_coeff'].original_args = [PrRa_coeff_initial]
                    self.estimator.problem.parameters['Pr_coeff'].args = [Pr_coeff_initial]
                    self.estimator.problem.parameters['PrRa_coeff'].args = [PrRa_coeff_initial]

                else:

                    # Start with the old estimates
                    Pr_est = self.truth.problem.parameters['Pr'] + self.estimator.problem.parameters['Pr_coeff'].args[0]
                    Ra_est = (self.truth.problem.parameters['Pr']*self.truth.problem.parameters['Ra'] + self.estimator.problem.parameters['PrRa_coeff'].args[0])/Pr_est

                    # Crank-Nicholson integration for relaxation equation
                    Pr_est = ((1 - 0.5*self.estimator.alpha*self.dt)*Pr_est + self.estimator.alpha*self.dt*new_Pr_est)/(1 + 0.5*self.estimator.alpha*self.dt)
                    Ra_est = ((1 - 0.5*self.estimator.alpha*self.dt)*Pr_est + self.estimator.alpha*self.dt*new_Pr_est)/(1 + 0.5*self.estimator.alpha*self.dt)

                    # Calculate parameters which should be used
                    Pr_coeff = Pr_est - self.truth.problem.parameters['Pr']
                    PrRa_coeff = Pr_est*Ra_est - self.truth.problem.parameters['Pr']*self.truth.problem.parameters['Ra']

                    # Set parameters again
                    self.estimator.problem.parameters['Pr_coeff'].original_args = [Pr_coeff]
                    self.estimator.problem.parameters['PrRa_coeff'].original_args = [PrRa_coeff]
                    self.estimator.problem.parameters['Pr_coeff'].args = [Pr_coeff]
                    self.estimator.problem.parameters['PrRa_coeff'].args = [PrRa_coeff]

                print('new Pr_est: ', new_Pr_est)
                print('new Ra_est: ', new_Ra_est)
                
                if self.estimator.solver.iteration != 0:
                    print('relaxed Pr_est: ', Pr_est)
                    print('relaxed Ra_est: ', Ra_est)

                print('Pr_error: ', self.estimator.problem.parameters['Pr_coeff'].args[0])
                print('PrRa_error: ', self.estimator.problem.parameters['PrRa_coeff'].args[0])

                # Step the estimator
                self.estimator.solver.step(self.dt)

                # Update steps and dt history
                self.estimator.prev_state = [self.estimator.prev_state[-1], self.estimator.solver.state['zeta']]
                self.estimator.dt_hist = [self.estimator.dt_hist[-1], self.dt]

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

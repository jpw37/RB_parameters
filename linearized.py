# unified_pr.py
"""
New approach to data assimilation and parameter recovery for convection:
combining the Boussinesq and nudging equations in a single dedalus problem.

Author: Jacob Murri
Creation Date: 2022-05-05
"""
# Utilities
from matplotlib import pyplot as plt
import numpy as np
import os
import time
import h5py
from scipy.integrate import simps
from scipy.special import factorial
from matplotlib.animation import writers as mplwriters
try:
    from tqdm import tqdm
except ImportError:
    print("Recommended: install tqdm (pip install tqdm)")
    tqdm = lambda x: x
from functools import partial

# Dedalus imports
from dedalus import public as de
from dedalus.extras import flow_tools
from dedalus.core.operators import GeneralFunction

# Other files
from base_simulator import BaseSimulator, RANK, SIZE
from RB_2D import RB_2D, P_N
from unified import RB_2D_DA

class RB_2D_PR_Linearized(RB_2D_DA):
    """
    Manager for dedalus simulations of data assimilation and parameter recovery
    for 2D Rayleigh-Benard convection.
    """

    def __init__(self, L=4., xsize=384, zsize=192, Prandtl=1., Rayleigh=1e6,
                 mu=1000., N=8, BCs="no-slip", Pr_guess=1., Ra_guess=1e6,
                 alpha=1., PrRa_RHS=False, nudge_T=False,
                 zeta_projection=P_N, T_projection=None, **kwargs):
        """
        Set up the systems of equations as a dedalus Initial Value Problem,
        without defining initial conditions.

        Parameters:
            L (float): the length of the x domain. In x and z, the domain is
                therefore [0,L]x[0,1]
            xsize (int): the number of points to discretize in the x direction
            zsize (int): the number of points to discretize in the z direction
            Prandtl (float): the ratio of momentum diffusivity to
                thermal diffusivity of the fluid
            Rayleigh (float): measures the amount of heat transfer due to
                convection, as opposed to conduction
            mu (float): constant on the Fourier projection in the
                Data Assimilation system
            N (int): the number of modes to keep in the Fourier projection
            BCs (str): if 'no-slip', use the no-slip BCs u(z=0,1) = 0.
                If 'free-slip', use the free-slip BCs u_z(z=0,1) = 0.
            PrRa_RHS (bool): if true, puts the Pr and Ra terms on the right hand
                side of the evolution equations. If not, formulates the estimated
                Pr and Ra as true value + error (with true on LHS, error on RHS)
            nudge_T (bool): if true, use nudging on the temperature variable as
                well (not implemented yet)

        """
        # Create an instance of the BaseSimulator class
        BaseSimulator.__init__(self, **kwargs)
        self.logger.info("BaseSimulator constructed")

        # Initialize domains
        x_basis = de.Fourier('x', xsize, interval=(0, L), dealias=3/2)
        z_basis = de.Chebyshev('z', zsize, interval=(0, 1), dealias=3/2)
        domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)

        # Initialize the problem as an IVP
        self.varlist = ['T', 'Tz', 'psi', 'psiz', 'zeta', 'zetaz', 'T_', 'Tz_',
                        'psi_', 'psiz_', 'zeta_', 'zetaz_']
        self.problem = de.IVP(domain, variables=self.varlist)

        # Set up remaining parameters
        self.setup_params(L=L, xsize=xsize, zsize=zsize, Prandtl=Prandtl,
                          Rayleigh=Rayleigh, mu=mu, N=N, Pr_guess=Pr_guess,
                          Ra_guess=Ra_guess, alpha=alpha, **kwargs)
        self.logger.info("Parameters set up")

        # Initialize auxiliary equations and BCs
        self.setup_auxiliary(BCs=BCs, **kwargs)
        self.logger.info("Auxiliary equations and BCs set up")

        # Initialize evolution equations
        self.setup_evolution()
        self.logger.info("Evolution equations constructed")

        # Save system parameters in JSON format.
        if RANK == 0: self._save_params()

    def setup_params(self, L, xsize, zsize, Prandtl, Rayleigh, mu, N, Pr_guess,
                     Ra_guess, alpha, **kwargs):
        """
        Sets up the parameters for the dedalus IVP problem. Does not set up the
        equations for the problem yet. Assumes self.problem is already defined.

        Parameters:
            L (float): the length of the x domain. In x and z, the domain is
                therefore [0,L]x[0,1]
            xsize (int): the number of points to discretize in the x direction
            zsize (int): the number of points to discretize in the z direction
            Prandtl (None or float): the ratio of momentum diffusivity to
                thermal diffusivity of the fluid. If None (default), then
                the system is set up as if Prandtl = infinity.
            Rayleigh (float): measures the amount of heat transfer due to
                convection, as opposed to conduction.

        """
        # Domain parameters
        self.problem.parameters['L'] = L
        self.problem.parameters['xsize'] = xsize
        self.problem.parameters['zsize'] = zsize

        # Fluid parameters
        self.problem.parameters['Ra'] = Rayleigh
        self.problem.parameters['PrInv'] = 1/Prandtl

        # Driving parameters
        self.problem.parameters['mu'] = mu
        self.mu = mu
        self.problem.parameters['N'] = N
        self.N = N

        # GeneralFunction for driving
        self.problem.parameters["driving_zeta"] = GeneralFunction(self.problem.domain, 'g', P_N, args=[])
        self.problem.parameters["driving_T"] = GeneralFunction(self.problem.domain, 'g', P_N, args=[])

        # Parameter estimation
        self.problem.parameters['PrInv_'] = GeneralFunction(self.problem.domain, 'g', self.const_val, args=[])
        self.problem.parameters['Ra_'] = GeneralFunction(self.problem.domain, 'g', self.const_val, args=[])
        self.InvPr_guess = 1/Pr_guess
        self.Ra_guess = Ra_guess


    def setup_evolution(self):
        """
        Set up the Boussinesq evolution equation and nudging equation for
        data assimilation.
        """

        # Boussinesq evolution equations
        self.problem.add_equation("Ra*dx(T) + dx(dx(zeta)) + dz(zetaz)  - dt(zeta) = v*dx(zeta) + w*zetaz")
        self.problem.add_equation("-dt(T) + PrInv*(dx(dx(T)) + dz(Tz)) = v*dx(T) + w*Tz")

        self.problem.add_equation("dx(dx(zeta_)) + dz(zetaz_) - dt(zeta_) = v_*dx(zeta_) + w_*zetaz_ - Ra_*dx(T_) - mu*driving_zeta")
        self.problem.add_equation("-dt(T_) + PrInv*(dx(dx(T_)) + dz(Tz_))= v_*dx(T_)+w_*Tz_  - mu*driving_T")

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

    def new_params(self):
        """

        """

        # Set scales
        self.solver.state['zeta'].set_scales(3/2)
        self.solver.state['zeta_'].set_scales(3/2)

        # Projections of errors
        proj_zeta_err = self.problem.domain.new_field()
        proj_zeta_err.set_scales(3/2)
        proj_zeta_err['g'] = P_N(self.solver.state['zeta']-self.solver.state['zeta_'], self.N)

        proj_T_err = self.problem.domain.new_field()
        proj_T_err.set_scales(3/2)
        proj_T_err['g'] = P_N(self.solver.state['T']-self.solver.state['T_'], self.N)

        # Important terms
        Ih_temp_x = self.problem.domain.new_field()
        Ih_temp_x.set_scales(3/2)
        Ih_temp_x['g'] = P_N(self.solver.state['T'].differentiate(x=1), self.N)

        Ih_laplace_temp_= self.problem.domain.new_field()
        Ih_laplace_temp_.set_scales(3/2)
        Ih_laplace_temp_['g'] = P_N(self.solver.state['T_'].differentiate(x=2)+self.solver.state['T_'].differentiate(z=2), self.N)

        # Calculate adjustments
        adjust_Ra = self.mu*de.operators.integrate(proj_zeta_err**2, 'x', 'z')['g'][0,0]/de.operators.integrate(Ih_temp_x*proj_zeta_err, 'x', 'z')['g'][0,0]
        adjust_PrInv = self.mu*de.operators.integrate(proj_T_err**2, 'x', 'z')['g'][0,0]/de.operators.integrate(Ih_laplace_temp_*proj_T_err, 'x', 'z')['g'][0,0]

        return float(self.problem.parameters['PrInv_'].args[0]-adjust_PrInv), float(self.problem.parameters['Ra_'].args[0]-adjust_Ra)

    def setup_simulation(self, scheme=de.timesteppers.RK443, sim_time=0.15, wall_time=np.inf, stop_iteration=np.inf, tight=False,
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
            if save_tasks is None: save_tasks = self.varlist

            # Add save tasks to list
            for task in save_tasks: self.snaps.add_task(task)

        # Convergence analysis ------------------------------------------------
        if analysis:
            # Save specific tasks in analysis/ files every few iterations.
            self.annals = self.solver.evaluator.add_file_handler(
                                    os.path.join(self.records_dir, "analysis"),
                                    sim_dt=save, max_writes=73600, mode="append") # iters = 20




            # Default analysis tasks
            if analysis_tasks is None:
                analysis_tasks = [
                                  ("1 + integ(w*T , 'x', 'z')/L", "Nu_1"),
                                  ("integ(dx(T)**2 + Tz**2, 'x', 'z')/L", "Nu_2"),
                                  ("integ(dx(v)**2 + dz(v)**2 + dx(w)**2 + dz(w)**2, 'x', 'z')", "Nu_3"),
                                  ("1 + integ(w_*T_ , 'x', 'z')/L", "Nu_1_da"),
                                  ("integ(dx(T_)**2 + Tz_**2, 'x', 'z')/L", "Nu_2_da"),
                                  ("integ(dx(v_)**2 + dz(v_)**2 + dx(w_)**2 + dz(w_)**2, 'x', 'z')", "Nu_3_da"),
                                  ("sqrt(integ(T**2, 'x', 'z'))", "T_L2"),
                                  ("sqrt( integ(dx(T)**2 + dz(T)**2, 'x', 'z'))", "gradT_L2"),
                                  ("sqrt( integ(v**2 + w**2, 'x', 'z'))", "u_L2"),
                                  ("sqrt( integ(dx(v)**2 + dz(v)**2 + dx(w)**2 + dz(w)**2, 'x', 'z'))", "gradu_L2"),
                                  ("sqrt( integ(dx(dx(T))**2 + dx(dz(T))**2 + dz(dz(T))**2, 'x', 'z'))", "T_h2"),
                                  ("sqrt(integ( dx(dx(v))**2 + dz(dz(v))**2 + dx(dz(v))**2 + dx(dz(w))**2 + dx(dx(w))**2 + dz(dz(w))**2, 'x','z'))", "u_h2"),
                                  ("P_N( zeta )", "Proj"),
                                  ("sqrt(integ((T-T_)**2, 'x', 'z'))", "T_err"),
                                  ("sqrt(integ(dx(T-T_)**2+dz(T-T_)**2, 'x', 'z'))", "gradT_err"),
                                  ("sqrt(integ((v-v_)**2 + (w-w_)**2, 'x', 'z'))", "u_err"),
                                  ("sqrt(integ(dx(v-v_)**2 + dz(v-v_)**2 + dx(w-w_)**2 + dz(w-w_)**2, 'x', 'z'))", "gradu_err"),
                                  ("sqrt( integ(dx(dx(T-T_))**2 + dx(dz(T-T_))**2 + dz(dz(T-T_))**2, 'x', 'z'))", "T_h2_err"),
                                  ("sqrt(integ( dx(dx(v-v_))**2 + dz(dz(v-v_))**2 + dx(dz(v-v_))**2 + dx(dz(w))**2 + dx(dx(w))**2 + dz(dz(w))**2, 'x','z'))", "u_h2_err"),
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

        # Define args for driving parameter
        self.zeta = self.solver.state['zeta']
        self.zeta.set_scales(1)
        self.zeta_ = self.solver.state['zeta_']
        self.zeta_.set_scales(1)
        self.dzeta = self.problem.domain.new_field(name='dzeta')
        self.dzeta['g'] = self.zeta['g'] - self.zeta_['g']

        # Substitute this projection for the "driving" parameter in the assimilating system
        self.problem.parameters["driving_zeta"].original_args = [self.dzeta, self.N]
        self.problem.parameters["driving_zeta"].args = [self.dzeta, self.N]

        # Define args for driving parameter
        self.T = self.solver.state['T']
        self.T.set_scales(1)
        self.T_ = self.solver.state['T_']
        self.T_.set_scales(1)
        self.dT = self.problem.domain.new_field(name='dT')
        self.dT['g'] = self.T['g'] - self.T_['g']

        # Substitute this projection for the "driving" parameter in the assimilating system
        self.problem.parameters["driving_T"].original_args = [self.dT, self.N]
        self.problem.parameters["driving_T"].args = [self.dT, self.N]

        # Set solver attributes
        self.solver.stop_sim_time = sim_time
        self.solver.stop_wall_time = wall_time
        self.solver.stop_iteration = stop_iteration

        # Set a flag
        self.solver_setup = True

    def setup_ic(self, initial_conditions):
        """
        Loads initial conditions from a filename.

        initial_conditions (None, str): determines from what source to
            draw the initial conditions. Valid options are as follows:
            - None: use trivial conditions (T_ = 1 - z, T = 1 - z + eps).
            - 'resume': use the most recent state file in the
                records directory (load both model and DA system).
            - An .h5 filename: load state variables for the model and
                reset the data assimilation state variables to zero.
        """
        # Initial conditions --------------------------------------------------
        if initial_conditions is None:

            # "Trivial" conditions.
            eps = 1e-4
            k = 3.117

            # Time step
            self.dt = 1e-8

            # Get grids from problem
            x, z = self.problem.domain.grids(scales=1)

            # Get temperature field
            T = self.solver.state['T']

            # Start T from rest plus a small perturbation
            T['g']  = 1 - z + eps*np.sin(k*x)*np.sin(2*np.pi*z)

            T.differentiate('z', out=self.solver.state['Tz'])

            self.logger.info("Using trivial initial conditions")

        elif initial_conditions == 'test':

            # Initial time step
            self.dt = 1e-8

            # "Trivial" conditions.
            eps = 1e-4
            k = 3.117

            # Grids
            x, z = self.problem.domain.grids(scales=1)

            for task in self.varlist:

                var = self.solver.state[task]

                if task == 'T':

                    var['g'] = 1 - z + 0.5*np.sin(k*x)*np.sin(2*np.pi*z)

                elif task == 'T_':

                    var['g'] = 1 - z + 0.5*np.sin(k*x)*np.sin(2*np.pi*z)

                else:

                    var['g'] = 0

        elif isinstance(initial_conditions, tuple):

            with h5py.File(initial_conditions[0], 'r') as infile:

                # Initial time step
                self.dt = 1e-8

                for name in ["T", "Tz", "psi", "psiz", "zeta", "zetaz"]:

                    # Get data
                    data = infile["tasks/"+name][-1,:,:]

                    # Determine the chunk belonging to this process.
                    chunk = data.shape[1] // SIZE
                    subset = data[:,RANK*chunk:(RANK+1)*chunk]

                    # Change the corresponding state variable.
                    scale = self.solver.state[name]['g'].shape[0] / \
                                        self.problem.parameters["xsize"]
                    self.solver.state[name].set_scales(data.shape[0]/self.problem.parameters["xsize"])#JW my ad-hoc new resolution restart
                    #solver.state[name].set_scales(1)
                    self.solver.state[name]['g'] = subset
                    self.solver.state[name].set_scales(scale)

            with h5py.File(initial_conditions[1], 'r') as infile:

                for name in ["T_", "Tz_", "psi_", "psiz_", "zeta_", "zetaz_"]:

                    # Get data
                    data = infile["tasks/"+name[:-1]][-1,:,:]

                    # Determine the chunk belonging to this process.
                    chunk = data.shape[1] // SIZE
                    subset = data[:,RANK*chunk:(RANK+1)*chunk]

                    # Change the corresponding state variable.
                    scale = self.solver.state[name]['g'].shape[0] / \
                                        self.problem.parameters["xsize"]
                    self.solver.state[name].set_scales(data.shape[0]/self.problem.parameters["xsize"])#JW my ad-hoc new resolution restart
                    #solver.state[name].set_scales(1)
                    self.solver.state[name]['g'] = subset
                    self.solver.state[name].set_scales(scale)

        elif isinstance(initial_conditions, str):   # Load data from a file.
            # Resume: load the state of the last (merged) state file.
            resume = initial_conditions == "resume"
            if resume:
                initial_conditions = self._get_merged_file("states")
            if not initial_conditions.endswith(".h5"):
                raise ValueError("'{}' is not an h5 file".format(
                                                        initial_conditions))
            # Load the data from the specified h5 file into the system.
            self.logger.info("Loading initial conditions from {}".format(
                                                        initial_conditions))

            with h5py.File(initial_conditions, 'r') as infile:
                self.dt = infile["scales/timestep"][-1] * .001    # JW: initial dt
#                dt = infile["scales/timestep"][-1] * .01    # initial dt
                errs = []
                tasks = ["T", "Tz", "psi", "psiz", "zeta", "zetaz"]
                if resume:      # Only load assimilating variables to resume.
                    tasks += ["T_", "Tz_", "psi_", "psiz_", "zeta_", "zetaz_"]
                    self.solver.sim_time = infile["scales/sim_time"][-1]
                    niters = infile["scales/iteration"][-1]
                    self.solver.initial_iteration = niters
                    self.solver.iteration = niters
                for name in tasks:
                    # Get task data from the h5 file (recording failures).
                    try:
                        data = infile["tasks/"+name][-1,:,:]
                    except KeyError as e:
                        errs.append("tasks/"+name)
                        continue
                    # Determine the chunk belonging to this process.
                    chunk = data.shape[1] // SIZE
                    subset = data[:,RANK*chunk:(RANK+1)*chunk]
                    # Change the corresponding state variable.
                    scale = self.solver.state[name]['g'].shape[0] / \
                                        self.problem.parameters["xsize"]
                    self.solver.state[name].set_scales(data.shape[0]/self.problem.parameters["xsize"])#JW my ad-hoc new resolution restart
                    #solver.state[name].set_scales(1)
                    self.solver.state[name]['g'] = subset
                    self.solver.state[name].set_scales(scale)
                if errs:
                    raise KeyError("Missing keys in '{}': '{}'".format(
                                    initial_conditions, "', '".join(errs)))

        # Initial conditions for assimilating system: T_0 = P_4(T0).
        # if not resume:
        #    G = self.problem.domain.new_field()
        #    G['c'] = solver.state['T']['c'].copy()
        #    solver.state['T_']['g'] = BoussinesqDataAssimilation2D.P_N(
        #                                                        G, 4, True)
        #    solver.state['T_'].differentiate('z', out=solver.state['Tz_'])


    def run_simulation(self):
        """
        Runs the simulation defined in self.setup_simulation, and then merges
        the results using self.merge_results.
        """

        assert self.solver_setup, 'Must run .setup_simulation first'

        # Running the simulation ----------------------------------------------

        try:
            # Start the counter
            self.logger.info("Starting simulation")
            start_time = time.time()

            # Iterate
            while self.solver.ok:

                # Use CFL condition to compute time step
                self.dt = self.cfl.compute_dt()

                if self.solver.iteration != 0:

                    print(f'Entering iteration {self.solver.iteration}; dt = {self.dt};, time = {self.solver.sim_time}')

                    plt.imshow(np.rot90(self.solver.state['zeta']['g']), cmap='cividis')
                    plt.title(f'True state at iteration {self.solver.iteration}')
                    plt.colorbar()
                    plt.axis('off')
                    plt.show()

                    plt.imshow(np.rot90(self.solver.state['T']['g']), cmap='coolwarm', vmax=1, vmin=0)
                    plt.title(f'True state at iteration {self.solver.iteration}')
                    plt.colorbar()
                    plt.axis('off')
                    plt.show()

                    plt.imshow(np.rot90(self.solver.state['zeta_']['g']), cmap='cividis')
                    plt.title(f'Assimilating state at iteration {self.solver.iteration}')
                    plt.colorbar()
                    plt.axis('off')
                    plt.show()

                    plt.imshow(np.rot90(self.solver.state['T']['g']), cmap='coolwarm',  vmax=1, vmin=0)
                    plt.title(f'Assimilating state at iteration {self.solver.iteration}')
                    plt.colorbar()
                    plt.axis('off')
                    plt.show()

                # Update parameters: different on first iteration than subsequent iterations
                if self.solver.iteration == 0:

                    # Use initial guesses
                    PrInv_est, Ra_est = self.InvPr_guess, self.Ra_guess

                elif self.solver.iteration % 50 == 0:

                    # Get update
                    PrInv_est, Ra_est = self.new_params()

                    print("New Rayleigh: ", Ra_est)


                # Set parameters
                self.problem.parameters['PrInv_'].original_args = [PrInv_est]
                self.problem.parameters['Ra_'].original_args = [Ra_est]
                self.problem.parameters['PrInv_'].args = [PrInv_est]
                self.problem.parameters['Ra_'].args = [Ra_est]

                # Define args for driving parameter
                self.zeta = self.solver.state['zeta']
                self.zeta.set_scales(1)
                self.zeta_ = self.solver.state['zeta_']
                self.zeta_.set_scales(1)
                self.dzeta = self.problem.domain.new_field(name='dzeta')
                self.dzeta['g'] = self.zeta['g'] - self.zeta_['g']

                # Substitute this projection for the "driving" parameter in the assimilating system
                self.problem.parameters["driving_zeta"].original_args = [self.dzeta, self.N]
                self.problem.parameters["driving_zeta"].args = [self.dzeta, self.N]

                # Define args for driving parameter
                self.T = self.solver.state['T']
                self.T.set_scales(1)
                self.T_ = self.solver.state['T_']
                self.T_.set_scales(1)
                self.dT = self.problem.domain.new_field(name='dT')
                self.dT['g'] = self.T['g'] - self.T_['g']

                # Substitute this projection for the "driving" parameter in the assimilating system
                self.problem.parameters["driving_T"].original_args = [self.dT, self.N]
                self.problem.parameters["driving_T"].args = [self.dT, self.N]

                # Step
                self.solver.step(self.dt)

                # Record properties every tenth iteration
                if self.solver.iteration % 10 == 0:

                    # Calculate max Reynolds number
                    Re = self.flow.max("Re")

                    # Output diagnostic info to log
                    info = "Iteration {:>5d}, Time: {:.7f}, dt: {:.2e}, Max Re = {:f}".format(
                        self.solver.iteration, self.solver.sim_time, self.dt, Re)
                    self.logger.info(info)

                    if np.isnan(Re):
                        raise ValueError("Reynolds number went to infinity!!"
                                         "\nRe = {}", format(Re))
        except BaseException as e:
            self.logger.error("Exception raised, triggering end of main loop.")
            raise

        finally:
            total_time = time.time()-start_time
            cpu_hr = total_time/60/60*SIZE
            self.logger.info("Iterations: {:d}".format(self.solver.iteration))
            self.logger.info("Sim end time: {:.3e}".format(self.solver.sim_time))
            self.logger.info("Run time: {:.3e} sec".format(total_time))
            self.logger.info("Run time: {:.3e} cpu-hr".format(cpu_hr))
            self.logger.debug("END OF SIMULATION")

            self.merge_results()

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

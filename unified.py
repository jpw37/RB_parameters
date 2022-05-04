# unified.py
"""
New approach to data assimilation and parameter recovery
for convection: combining all of the equations in a single
dedalus problem.

Author: Jacob Murri
Creation Date: 2022-05-04
"""
# Utilities
import numpy as np
import os
import time

# Dedalus imports
from dedalus import public as de
from dedalus.extras import flow_tools
from dedalus.core.operators import GeneralFunction

# Other files
from base_simulator import BaseSimulator, RANK, SIZE
from RB_2D import RB_2D, P_N



class RB_2D_DA(RB_2D):
    """
    Manager for dedalus simulations of data assimilation for 2D Rayleigh-
    Benard convection.
    """

    def __init__(self, L=4, xsize=384, zsize=192, Prandtl=1, Rayleigh=1e6, mu=1000.,
                 N=8, BCs="no-slip", **kwargs):
        """
        Set up the systems of equations as a dedalus Initial Value Problem,
        without defining initial conditions.

        Parameters:
            L (float): the length of the x domain. In x and z, the domain is
                therefore [0,L]x[0,1].
            xsize (int): the number of points to discretize in the x direction.
            zsize (int): the number of points to discretize in the z direction.
            Prandtl (float): the ratio of momentum diffusivity to
                thermal diffusivity of the fluid
            Rayleigh (float): measures the amount of heat transfer due to
                convection, as opposed to conduction.
            mu (float): constant on the Fourier projection in the
                Data Assimilation system.
            N (int): the number of modes to keep in the Fourier projection.
            BCs (str): if 'no-slip', use the no-slip BCs u(z=0,1) = 0.
                If 'free-slip', use the free-slip BCs u_z(z=0,1) = 0.

        """
        # Create an instance of the BaseSimulator class
        BaseSimulator.__init__(self, **kwargs)
        self.logger.info("BaseSimulator constructed")

        # Initialize domains
        x_basis = de.Fourier('x', xsize, interval=(0, L), dealias=3/2)
        z_basis = de.Chebyshev('z', zsize, interval=(0, 1), dealias=3/2)
        domain = de.Domain([x_basis, z_basis], grid_dtype=np.float64)

        # Initialize the problem as an IVP
        self.varlist = ['T', 'Tz', 'psi', 'psiz', 'zeta', 'zetaz', 'T_', 'Tz_', 'psi_', 'psiz_', 'zeta_', 'zetaz_']
        self.problem = de.IVP(domain, variables=self.varlist)

        # Set up remaining parameters
        self.setup_params(L=L, xsize=xsize, zsize=zsize, Prandtl=Prandtl, Rayleigh=Rayleigh, mu=mu, N=N, **kwargs)
        self.logger.info("Parameters set up")

        # Initialize auxiliary equations and BCs
        self.setup_auxiliary(BCs=BCs, **kwargs)
        self.logger.info("Auxiliary equations and BCs set up")

        # Initialize evolution equations
        self.setup_evolution()
        self.logger.info("Evolution equations constructed")

        # Save system parameters in JSON format.
        if RANK == 0: self._save_params()

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

    def setup_auxiliary(self, BCs, **kwargs):
        """
        Assuming that the dedalus IVP problem is contained in self.problem, defines
        boundary conditions and auxiliary (non-time-dependent) equations for the problem.

        Parameters:
            BCs (str): if 'no-slip', use the no-slip BCs u(z=0,1) = 0.
                If 'free-slip', use the free-slip BCs u_z(z=0,1) = 0.
        """

        # Add Boussinesq equations --------------------------------------------

        # Stream function substitutions: u = [v, w] = [-psi_z, psi_x]
        self.problem.substitutions['v']  = "-psiz" # or -dz()
        self.problem.substitutions['w']  =  "dx(psi)"
        self.problem.substitutions['v_']  = "-psiz_" # or -dz()
        self.problem.substitutions['w_']  =  "dx(psi_)"

        # zeta = laplace(psi)
        self.problem.add_equation("zeta  - dx(dx(psi))  - dz(psiz)  = 0")
        self.problem.add_equation("zeta_  - dx(dx(psi_))  - dz(psiz_)  = 0")

        # Relate higher-order z derivatives (b/c Chebyshev).
        self.problem.add_equation("psiz  - dz(psi)  = 0")
        self.problem.add_equation("zetaz  - dz(zeta)  = 0")
        self.problem.add_equation("Tz  - dz(T)  = 0")
        self.problem.add_equation("psiz_  - dz(psi_)  = 0")
        self.problem.add_equation("zetaz_  - dz(zeta_)  = 0")
        self.problem.add_equation("Tz_  - dz(T_)  = 0")

        # Boundary Conditions -------------------------------------------------

        # Other boundary conditions not implemented yet
        assert BCs in ['no-slip', 'free-slip'], "'BCs' must be 'no-slip' or 'free-slip'"

        # Temperature heating from 'left' (bottom), cooling from 'right' (top).
        self.problem.add_bc("left(T)  = 1")          # T(z=0) = 1
        self.problem.add_bc("right(T)  = 0")         # T(z=1) = 0
        self.problem.add_bc("left(T_)  = 1")          # T(z=0) = 1
        self.problem.add_bc("right(T_)  = 0")

        # Velocity field boundary conditions: no-slip or free-slip.
        # w(z=1) = w(z=0) = 0 (part of no-slip and free-slip)
        self.problem.add_bc("left(psi)  = 0")
        self.problem.add_bc("right(psi)  = 0")
        self.problem.add_bc("left(psi_)  = 0")
        self.problem.add_bc("right(psi_)  = 0")

        if BCs == "no-slip":

            # u(z=0) = 0 --> v(z=0) = 0 = w(z=0)
            self.problem.add_bc("left(psiz)  = 0")
            self.problem.add_bc("left(psiz_)  = 0")

            # u(z=1) = 0 --> v(z=1) = 0 = w(z=1)
            self.problem.add_bc("right(psiz)  = 0")
            self.problem.add_bc("right(psiz_)  = 0")

        elif BCs == "free-slip":
            # u'(z=0) = 0 --> v'(z=0) = 0 = w(z=0)
            self.problem.add_bc("left(dz(psiz))  = 0")
            self.problem.add_bc("left(dz(psiz_))  = 0")

            # u'(z=1) = 0 --> v'(z=1) = 0 = w(z=1)
            self.problem.add_bc("right(dz(psiz))  = 0")
            self.problem.add_bc("right(dz(psiz_))  = 0")

    def setup_evolution(self):
        """
        Set up the Boussinesq evolution equation and nudging equation for
        data assimilation.
        """

        # Boussinesq evolution equations
        self.problem.add_equation("Pr*(Ra*dx(T) + dx(dx(zeta)) + dz(zetaz))  - dt(zeta) = v*dx(zeta) + w*zetaz")
        self.problem.add_equation("dt(T) - dx(dx(T)) - dz(Tz) = - v*dx(T) - w*Tz")

        # Nudging equations
        self.problem.add_equation("Pr*(Ra*dx(T_) + dx(dx(zeta_)) + dz(zetaz_)) - dt(zeta_) = v_*dx(zeta_) + w_*zetaz_ - mu*driving")
        self.problem.add_equation("dt(T_) - dx(dx(T_)) - dz(Tz_) = -v_*dx(T_) - w_*Tz_")

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
                                  ("sqrt(integ(T**2, 'x', 'z'))", "T_L2"),
                                  ("sqrt( integ(dx(T)**2 + dz(T)**2, 'x', 'z'))", "gradT_L2"),
                                  ("sqrt( integ(v**2 + w**2, 'x', 'z'))", "u_L2"),
                                  ("sqrt( integ(dx(v)**2 + dz(v)**2 + dx(w)**2 + dz(w)**2, 'x', 'z'))", "gradu_L2"),
                                  ("sqrt( integ(dx(dx(T))**2 + dx(dz(T))**2 + dz(dz(T))**2, 'x', 'z'))", "T_h2"),
                                  ("sqrt(integ( dx(dx(v))**2 + dz(dz(v))**2 + dx(dz(v))**2 + dx(dz(w))**2 + dx(dx(w))**2 + dz(dz(w))**2, 'x','z'))", "u_h2"),
                                  ("P_N( zeta )", "Proj"),
                                  ("sqrt(integ((T-T_)**2, 'x', 'z'))", "T_err_L2"),
                                  ("sqrt(integ((v-v_)**2 + (w-w_)**2, 'x', 'z'))", "u_err_L2")
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

        elif initial_conditions == 'conductive':

            # Initial time step
            self.dt = 1e-8

            # Grids
            x, z = self.problem.domain.grids(scales=1)

            for task in self.varlist:

                var = self.solver.state[task]

                if task[0] == 'T':

                    var['g'] = 1 - z

                else:

                    var['g'] = 0

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

                # Get projection information for Nudging
                # true state
                self.zeta = self.solver.state['zeta']
                self.zeta.set_scales(1)

                # assimilating state
                self.zeta_ = self.solver.state['zeta_']
                self.zeta_.set_scales(1)

                # Get projection of difference between assimilating state and true state
                self.dzeta = self.problem.domain.new_field(name='dzeta')
                self.dzeta['g'] = self.zeta['g'] - self.zeta_['g']

                # Substitute this projection for the "driving" parameter in the assimilating system
                if self.solver.iteration == 0: self.problem.parameters["driving"].original_args = [self.dzeta, self.N]
                self.problem.parameters["driving"].args = [self.dzeta, self.N]

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

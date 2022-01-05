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
from RB_2D import RB_2D
from RB_2D_assim import RB_2D_assim



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

    def const_val(self, value):
        """
        Assuming that the problem is already defined, create a new field on the
        problem's domain with constant value given by the argument value.

        Parameters:
            value (numeric): the value which will be given to the new field.
        """
        coefficient_field = self.problem.domain.new_field
        coefficient_field['g'] = value
        return coefficient_field #somehow need to make this so that the field isn't defined every time it is called, and the domain is available...maybe define a field at the initial step and then have that stored in self, then adjusted as necessary?

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
        self.problem.parameters['driving'] = GeneralFunction(self.problem.domain, 'g', P_N, args=[])

        # Parameter estimation
        problem.parameters['Pr_coeff'] = GeneralFunction(problem.domain, 'g', const_val, args=[])
        problem.parameters['PrRa_coeff'] = GeneralFunction(problem.domain, 'g', const_val, args=[])

        # Store a little history for estimation of backwards time derivative
        self.problem.parameters['Pr_coeff'].args = [0.]
        self.problem.parameters['Pr_coeff'].original_args = [0.]
        self.problem.parameters['PrRa_coeff'].args = [0.]
        self.problem.parameters['PrRa_coeff'].orginal_args = [0.]

        #Need to initialize the previous 2 states and corresponding time steps
        self.prev_state = [None] * 2
        self.dt_hist = [None] * 2

    def new_params(self, zeta):
        """Method to estimate the parameters at each step"""

        # Get estimated state and backward time derivative
        zeta_tilde = self.solver.state['zeta']
        zeta_t = self.backward_time_derivative()

        # set up the alpha_i coefficients
        laplace_zeta = zeta_tilde.differentiate(x=2)+zeta.differentiate(z=2)
        Ih_laplace_zeta = P_N(laplace_zeta, self.N)

        # set up the beta_i coefficients
        Ih_zeta_x = P_N(zeta_tilde.differentiate(x=1),self.N)

        # set up the gamma_i coefficients
        remainder = self.problem.domain.new_field
        nonlinear_term = self.problem.domain.new_field
        nonlinear_term['g'] = self.solver.state['u']['g']*zeta_tilde.differentiate(x=1)['g'] + self.solver.state['w']['g']*zeta_tilde.differentiate(z=1)['g']
        remainder['g'] = mu*(zeta['g'] - zeta_tilde['g']) - nonlinear_term['g'] - zeta_t['g']
        Ih_remainder = P_N(remainder,self.N)

        e1 = self.problem.domain.new_field
        e2 = self.problem.domain.new_field

        #set e1 to be the projection of the error, guaranteeing exponential decay of the error
        e1 = P_N(zeta-zeta_tilde,self.N)#JPW: check the order of this, should it be zeta_tilde-zeta?
        e2 = Ih_laplace_zeta #start with this choice...others are possible
        #JPW: use I_h(error) as e_1, then start with something already computed, i.e. laplace operator (probably not nonlinear term...stick with linear differential operator ie don't square anything).  Use modified Gram-Schmidt to force this to be orthogonal to e_1 to give the 2nd direction

        # now perform modified Gram-Schmidt on e1 and e2 (no need to normalize I don't think)
        e2['g'] = e2['g'] - de.operators.integrate(e1*e2,'x','z')*e1['g']

        # Now e1 and e2 should be orthogonal. Calculate the coefficients
        alpha1 = de.operators.integrate(e1*Ih_laplace_zeta, 'x', 'z')
        alpha2 = de.operators.integrate(e2*Ih_laplace_zeta, 'x', 'z')

        beta1 = de.operators.integrate(e1*Ih_zeta_x, 'x', 'z')
        beta2 = de.operators.integrate(e2*Ih_zeta_x, 'x', 'z')

        gamma1 = de.operators.integrate(e1*Ih_remainder, 'x', 'z')
        gamma2 = de.operators.integrate(e2*Ih_remainder, 'x', 'z')

        # Set up linear system and solve
        A = np.array([[alpha1, beta1], [alpha2, beta2]])
        b = np.array([[gamma1], [gamma2]])
        Pr, PrRa = np.linalg.solve(A,b)

        # Return estimated coefficients
        return Pr, PrRa/Pr

    def backward_time_derivative(self):
        """Calculuate the time derivative of the observed vorticity based off of the observed system.
        Derivation of the derivative approximation is made in the main document."""

        if None in self.dt_hist:
            # There may be a better way to do this.  Right now zeta_t=0 for the first 2 time steps
            zeta_t = self.const_val(0.)

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
            zeta_t = self.problem.domain.new_field
            zeta_t['g'] = c0*self.solver.state['zeta']['g'] + c1*self.prev_state[-1]['g'] + c2*self.prev_state[-2]

        return zeta_t

import numpy as np
from mpi4py import MPI
import h5py
import os
import glob

RANK = MPI.COMM_WORLD.rank        # Which process this is running on
SIZE = MPI.COMM_WORLD.size

class Resume(object):
    """
    A class which stores the state of each variable in a PDE at a given time,
    along with some valuable restarting information, like a time step.
    """

    def __init__(self, shape, varlist, dtype=np.float64):
        """
        Initialize a Resume class.

        Parameters:
            shape (tuple of int): shape of states for this problem
            varlist (list of str): list of names for variables which will be stored
            dtype (dtype): datatype of states

        Returns:
            Resume object
        """

        # Store grid shape and dtype
        self.shape = shape
        self.dtype = dtype

        # Create dictionary mapping variables to states
        self.states = { var:np.zeros(shape, dtype=dtype) for var in varlist }

        # Store metadata
        self.metadata = {'timestep': None, 'sim_time': 0., 'iteration': 0}

    def set_state(self, var, value):
        """
        Set the value of one grid.

        Parameters:
            var (str): variable name of grid to be set
            value (array-like or float): values to be set
        """

        # If the value is a float, set the whole grid to that constant
        if type(value) == float:

            self.states[var] = value*np.ones(self.shape, dtype=self.dtype)

        # If the value is array-like
        else:

            # Check to make sure shape is ok
            assert self.shape == value.shape

            # Set a grid
            self.states[var] = np.array(value, dtype=self.dtype)

    def set_state_many(self, d):
        """
        Quickly set multiple states.

        Parameters:
            d (dict): mapping variable names to values to be set.
        """

        # Iterate through the given dict
        for var, value in d.items():

            self.set_state(var, value)

    def get_state(self, var):
        """
        Get the value of a particular grid.

        Parameters:
            var (str): variable name of a grid

        Returns:
            (ndarray): state

        """

        return self.states[var]

    def get_varlist(self):
        """
        Returns variable list.

        Returns:
            (list of str) problem variables.
        """

        return list(self.states.keys())

    def set_metadata(self, name, data):
        """
        Set the timestep.

        Parameters:
            name (str): name of data to be set
            data (float > 0)
        """

        self.metadata[name] = data

    def get_metadata(self, name):
        """
        Returns a piece of metadata.

        returns:
            (float)
        """
        return self.metadata[name]

    def add_assim(self, assim):
        """
        Add assimilating state variables to this Resume object. Variables will
        be renamed to include suffix _ if not already present.

        Parameters:
            assim (Resume): PDE_Grid_State containing relevant variables.
        """

        # Add variables
        for var in assim.get_varlist():

            # Rename to include suffix _ if not already present.
            if var[-1] == '_':
                new_name = var
            else:
                new_name = var+'_'

            # Store grid
            self.set_state(new_name, assim.get_state(var))

        # Adjust timestep if necessary
        if type(assim.get_metadata('timestep')) == float and type(self.get_metadata('timestep')) == float:
             if assim.get_metadata('timestep') < self.get_metadata('timestep'):
                 self.set_metadata('timestep', assim.get_metadata('timestep'))

def temperature_sine_wave(vertical_periods=1, horizontal_periods=2, magnitude=0.5, xsize=384, zsize=192):
    """
    Create a Resume object with an initial temperature state that looks like T = 1 - x + c*sin(k*x)*sin(l*z).

    Parameters:
        vertical_periods (int/float):  number of horizontal periods for sin (l in formula above)
        horizontal_periods (int/float): number of horizontal periods for sin (k in formula above)
        magnitude (float): magnitude of sin perturbation
        xsize (int): zsize of grid
        zsize (int): zsize of grid

    Returns:
        Resume object (no timestep)
    """

    # Get grids
    x, z = np.linspace(0, 1, xsize).reshape((-1,1)), np.linspace(0, 1, zsize).reshape((1,-1))

    # Set up Resume object
    states = Resume(shape=(xsize, zsize), varlist = ['T', 'Tz', 'psi', 'psiz', 'zeta', 'zetaz'])

    # Set temperature state
    states.set_state('T', 1 - z + magnitude*np.sin(2*np.pi*horizontal_periods*x)*np.sin(2*np.pi*vertical_periods*z))

    return states


def start_from(filename='most recent', set_time=False, timestep_reduction=1, start_point=-1, pattern="RB_2D_[!ap]*"):

    if filename == 'most recent':

        # get list of files that matches pattern
        files = list(glob.glob(pattern))

        # sort by modified time
        files.sort(key=lambda x: os.path.getmtime(x))

        # get last item in list
        filename = files[-1] + '/states/states.h5'

    with h5py.File(filename, 'r') as infile:

        for i, var in enumerate(list(infile['tasks'].keys())):

            data = infile['tasks/'+var][start_point, :, :]
            print(data.shape)

            # Determine the chunk belonging to this process.
            # chunk = data.shape[1] // SIZE
            # subset = data[:,RANK*chunk:(RANK+1)*chunk]

            # Initialize resume object (if first iteration)
            if i == 0: ic = Resume(data.shape, list(infile['tasks'].keys()))

            ic.set_state(var, data)

        # Record timestep
        ic.set_metadata('timestep', infile["scales/timestep"][start_point]*timestep_reduction)

        # Reset counters if necessary
        if set_time:
            ic.set_metadata('iteration', infile["scales/iteration"][start_point])
            ic.set_metadata('sim_time', infile["scales/sim_time"][start_point])

    return ic

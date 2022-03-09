"""
Created on Fri Nov 2 12:02:09 2018

@author: azarketa

This file is intended to act as a side-module to scripts that aim at processing TDMS files. It includes the necessary classes and methods
to process such files that store MGEP's wind tunnel data.
"""

######################################################################################################################
########################################################PACKAGES######################################################
######################################################################################################################

# numpy, scipy and statistics are intended to perform numeric calculations.
import numpy as np
import scipy.integrate
# Importing matplotlib.pyplot for plotting purposes (uncomment if necessary).
import matplotlib.pyplot as plt

# The entities classmethod and staticmethod are imported from the builtins package as function or class decorators.
# The classmethod decorator provides an akin functionality to the C# overloading of a function.
# The staticmethod decorator provides an akin functionality to the C# static method, which is to say that the object
# containing the static method need not be instantiated in order to use that method; what's more, a static method does
# not provide access to the parent object nor to other internals.
from builtins import staticmethod

# nptdms is a package that provides basic functionalities for reading a TDMS file (https://pypi.org/project/npTDMS/).
import nptdms

# TDMS_Packages.TDMSEnums is a custom package containing customized Enum objects used on the workflow herein.
import own_packages.TDMS_Packages.TDMSEnums as TDMSE

# TDMS_Packages.MathTools is a custom package containing customaized mathematical functions.
import own_packages.Math_Tools.MathTools as mt

# The library dataclasses implements the functionality of the dataclass decorator, which is used to create C#-like
# structs.
import dataclasses as dtc

# The os library is intended to obtain information and perform operations with operative-system-related objects.
import os

# Serialization for saving files in binary format.
import pickle

# sys is intended to perform system-level operations, such as the searching and retrieval of the console frame.
import sys

# inspect is intended to perform 'reflection'-type operations (in C# terminology) (i.e. look for caller functions, for example).
import inspect

# termcolor.color to perform colored terminal prints.
from termcolor import colored

######################################################################################################################
###################################################CLASS HIERARCHY####################################################
######################################################################################################################

# The following hierarchy intends to reproduce the structure of a generic TDMS file. A generic TDMS file structure
# consists of a root entity and an entity containing a number of groups, with each group containing a number of
# channels:
#
# ROOT
#  |
#  |
#  --GROUPS
#      |
#      |
#      --GROUP_1
#          |
#          |
#          --CHANNEL_1
#          --CHANNEL_2
#               .
#               .
#               .
#          --CHANNEL_N
#      --GROUP_2
#          |
#          |
#          --CHANNEL_1
#          --CHANNEL_2
#               .
#               .
#               .
#          --CHANNEL_M
#          .
#          .
#          .
#      --GROUP_L
#          |
#          |
#          --CHANNEL_1
#          --CHANNEL_2
#               .
#               .
#               .
#          --CHANNEL_K
#
# The structure of a TDMS file is reproduced on a class-based fashion; the root entity (tdmsFileRoot) is the only
# 'public' class (on a C# terminology) accessible by external code. The other classes (groups-containing, 
# __tdmsFileGroups__; group-representing, __tdmsFileGroup__; channels-containing, __TdmsGroupChannels__; channel-
# representing, __TdmsGroupChannel__), are private internal classes (on C# terminology). As Python does not own an
# analogous construction for defining such classes, they are defined by notation convention (the double underscores
# before and aft the class declarations are meant, by convention, to warn potential coders and/or users that those
# classes are private by definition).

######################################################################################################################
########################################################MAIN CLASS####################################################
######################################################################################################################

# Public class tdmsFileRoot.
class TdmsFileRoot:
    """Public class tdmsFileRoot representing the core entity of a TDMS file."""

    # __init__ (constructor) of the public class tdmsFileRoot.
    def __init__(self, file_path, tdms_file_read_mode, file_alias, ref_mag, nd=True):
        """Initializes an instance of the tdmsFileRoot object.

        The tdmsFileRoot object is the basic object that contains the data of a TDMS file.        

        - **parameters**, **return**, **return types**::

        :param file_path: path of the file to be read.
        :param tdms_file_read_mode: custom enum object specifying the reading mode:
            -standard: reads and damps the TDMS structure, as a whole, to runtime variables.
            -means: reads and damps the TDMS structure to runtime variables, adding to the structure itself
            the average values of the read data.
            -means_only: reads and damps the TDMS structure to runtime variables, considering solely the
            average values of the read data.
            -projected: to be used when force measurements are included; it projects the forces according
            to the measurement angle to provide the loads on wind tunnel axes.
            -projected_means_only: to be used when force measurements are included; it projects the force
            averages according to the measurement angle to provide the average loads on wind tunnel axes.
            -wake_rake: to be used when wake rake measurements are included.
        :param file_alias: string by which the data pertaining to a specific file is recognized on further
        charts or tables.
        :param ref_mag: an instantiation of the dataclass RefMagnitudes for non-dimensionalizing purposes.
        :param nd: boolean flag; if True, the non-dimensionalizing method is run on the results.
        :return: tdmsFileRoot object structured in accordance to the provided reading mode.
        :rtype: TdmsFileRoot
        
        """
        
        # Calling Python's dynamic method lookup 'getattr' to get the proper initializer, thus avoiding the 
        # need of an exceedingly long 'if' tree.
        getattr(self, "__" + tdms_file_read_mode.name + "_init__")(file_path, file_alias, ref_mag, nd)


    #######################################################################################################
    ################################################INITIALIZERS###########################################
    #######################################################################################################

    # standard reading mode initializer.
    def __standard_init__(self, file_path, file_alias, ref_mag, nd=True):
        """Initializes an instance of the tdmsFileRoot object in "Standard" reading mode.

        The instantiation of a tdmsFileRoot object in "Standard" reading mode leads to the damping of the structure
        of the provided TDMS file, as a whole, to runtime variables.

        - **parameters**::
        
        :param file_path: path of the file to be read.
        :param file_alias: string by which the data pertaining to a specific file is recognized on further charts
        or tables.
        :param ref_mag: an instantiation of the dataclass RefMagnitudes for non-dimensionalizing purposes.
        :param nd: boolean flag; if True, the non-dimensionalizing method is run on the results.

        """

        # Sets the 'file_alias' parameter as the new attribute '__Alias__' of the tdmsFileRoot object being
        # instantiated.
        self.__setattr__('__alias__', file_alias)
        # Sets the 'file_path' parameter as the new attribute '__path__' of the tdmsFileroot object being instantiated.
        self.__setattr__('__path__', file_path)
        # Sets the 'nd_d' parameter as the new attribute '__nd_d__' of the tdmsFileRoot object being instantiated.
        # This parameter intends to store a boolean telling whether the data has been non-dimensionalized.
        self.__setattr__('__nd_d__', False)
        # Sets the 'unc_d' parameter as the new attribute '__unc_d__' of the tdmsFileRoot object being instantiated.
        # This parameter intends to store a boolean telling whether the data has been uncertainty-analysed.
        self.__setattr__('__unc_d__', False)
        # Calls the internal __setRootProperties__ function to set the root properties.
        self.__set_root_properties__(file_path)
        # Instantiates a new internal __tdmsFileGroups__ object that stores the group structure of the TDMS file, and
        # sets that group into the groups_original variable of the tdmsFileRoot object being instantiated.
        self.__setattr__("groups_original", __TdmsFileGroups__(self.__file__, self.__file__.groups(), generate_isolated_groups=False))        
        # Sets the list of RefMagnitude objects attending to the number of measurements included in the TDMS file.
        # Each of the measurements is meant to have its own referential magnitudes, which are computed further on.
        measurements = len([meas for meas in dir(self.groups_original) if "__" not in meas])
        self.__setattr__('__ref_mags__', [RefMagnitudes(length=ref_mag.length, span=ref_mag.span, thick=ref_mag.thick, vel=ref_mag.u, press=ref_mag.p, temp=ref_mag.temp, rh=ref_mag.rh, r_const=ref_mag.r_const) for _ in range(0, measurements)])
        # Conditional that, if True, calls the internal __dimensionalize__ method with the correspondent RefMagnitudes object
        # (referential magnitudes).
        if nd:
            self.__dimensionalize__(self.__file__)

    # means reading mode initializer.
    def __means_init__(self, file_path, file_alias, ref_mag, nd=True):
        """Initializes an instance of the tdmsFileRoot object in "Means" reading mode.

        The instantiation of a tdmsFileRoot object in "Means" reading mode leads to the damping of the
        structure of the provided TDMS file, in addition to the averages of the read data, to runtime variables.

        - **parameters**::

        :param file_path: path of the file to be read.
        :param file_alias: string by which the data pertaining to a specific file is recognized on further charts
        or tables.
        :param ref_mag: an instantiation of the dataclass RefMagnitudes for non-dimensionalizing purposes.
        :param nd: boolean flag; if True, the non-dimensionalizing method is run on the results.

        """

        # Calls the internal __standardInit__ method to set the structure of the TDMS file, as a whole, in the runtime
        # variable.
        self.__standard_init__(file_path, file_alias, ref_mag, nd)
        # Calls the internal __setMeans__ for setting the channel properties and their correspondent values into
        # variables contained within the means_channels variable.
        self.__set_means__()

    # means_only reading mode initializer.
    def __means_only_init__(self, file_path, file_alias, ref_mag, nd=True):
        """Initializes an instance of the tdmsFileRoot object in "Means" reading mode.

        The instantiation of a tdmsFileRoot object in "MeansOnly" reading mode leads to the damping of the structure
        of the provided TDMS file, considering solely the averaged values of the read data.

        - **parameters**::

        :param file_path: path of the file to be read.
        :param file_alias: string by which the data pertaining to a specific file is recognized on further charts
        or tables.
        :param ref_mag: an instantiation of the dataclass RefMagnitudes for non-dimensionalizing purposes.
        :param nd: boolean flag; if True, the non-dimensionalizing method is run on the results.

        """

        # Calls the internal __meansInit__ method to set the structure of the TDMS file, as a whole, in the runtime
        # variable.
        self.__means_init__(file_path, file_alias, ref_mag, nd)
        # Deleting the groups_original variable, with groups_added being the one remaining.
        if hasattr(self, 'groups_original'):
            self.__setattr__('__groups_original__', self.__getattribute__('groups_original'))
            self.__delattr__('groups_original')

    # projected reading mode initializer.
    def __projected_init__(self, file_path, file_alias, ref_mag, nd=True):
        """Initializes an instance of the tdmsFileRoot object in "Projected" reading mode.

        The instantiation of a tdmsFileRoot object in "Projected" reading mode is to be used when force measurements
        are included; it projects the forces according to the measurement angle to provide the loads on wind tunnel
        axes.

        - **parameters**::

        :param file_path: path of the file to be read.
        :param file_alias: string by which the data pertaining to a specific file is recognized on further charts
        or tables.
        :param ref_mag: an instantiation of the dataclass RefMagnitudes for non-dimensionalizing purposes.
        :param nd: boolean flag; if True, the non-dimensionalizing method is run on the results.

        """

        # This initializer undertakes a process similar to that of the means init method, which is why it relies on the
        # previously defined __meansInit__ method for performing the data damping.
        self.__means_init__(file_path, file_alias, ref_mag, nd)
        # Calls the internal __project_forces__ method for performing the projecting operation on the loads.
        self.__project_forces__(self.__file__, False)
        self.__set_kistler__()

    # projected_means_only reading mode initializer.
    def __projected_means_only_init__(self, file_path, file_alias, ref_mag, nd=True):
        """Initializes an instance of the tdmsFileRoot object in "ProjectedMeansOnly" reading mode.

        The instantiation of a tdmsFileRoot object in "ProjectedMeansOnly" reading mode is to be used when force
        measurements are included; it projects the averaged forces according to the measurement angle to provide the
        loads on wind tunnel axes.

        - **parameters**::

        :param file_path: path of the file to be read.
        :param file_alias: string by which the data pertaining to a specific file is recognized on further charts
        or tables.
        :param ref_mag: an instantiation of the dataclass RefMagnitudes for non-dimensionalizing purposes.
        :param nd: boolean flag; if True, the non-dimensionalizing method is run on the results.

        """

        # As this initializer performs a task similar to that of the __meansOnlyInit__ method, it relies on it to
        # undertake the data loading part.
        self.__means_only_init__(file_path, file_alias, ref_mag, nd)
        # Calls the internal __projectForces__ method for performing the projecting operation on the loads.
        self.__project_forces__(self.__file__, True)
        # Calls the internal __set_wake_rake__() method for performing the attribute structuring task.
        self.__set_kistler__()

    # wake_rake reading mode initializer.
    def __wake_rake_init__(self, file_path, file_alias, ref_mag, nd=True):
        """Initializes an instance of the tdmsFileRoot object in "wake_rake" reading mode.

        The instantiation of a tdmsFileRoot object in "wake_rake" reading mode is to be used when wake rake surveys
        are undertaken; it builds the attribute structure corresponding to the wake rake surveying mode.

        -**parameters**::

        :param file_path: path of the file to be read.
        :param file_alias: string by which the data pertaining to a specific file is recognized on further charts
        or tables.
        :param ref_mag: an instantiation of the dataclass RefMagnitudes for non-dimensionalizing purposes.
        :param nd: boolean flag; if True, the non-dimensionalizing method is run on the results.

        """

        # As this initializer performs a task similar to that of the __means_only_init__ method, it relies on it to
        # undertake the data loading part.
        self.__means_only_init__(file_path, file_alias, ref_mag, nd)
        # Calls the internal __set_wake_rake__() method for performing the attribute structuring task.
        self.__set_wake_rake__()
        
    # cobra reading mode initializer.
    def __cobra_init__(self, file_path, file_alias, ref_mag, nd=True):
        """Initializes an instance of the tdmsFileRoot object in "cobra" reading mode.
        
        The instantiation of a tdmsFileRoot object in "cobra" reading mode is to be used when Cobra probe surveys
        are undertaken; it builds the attribute structure corresponding to the wake rake surveying mode.
        
        -**parameters**::
            
        :param file_path: path of the file to be read.
        :param file_alias: string by which the data pertaining to a specific file is recognized on further charts
        or tables.
        :param ref_mag: an instantiation of the dataclass RefMagnitudes for non-dimensionalizing purposes.
        :param nd: boolean flag; if True, the non-dimensionalizing method is run on the results.
        
        """
  
        # As this initializer performs a task similar to that of the __means_only_init__ method, it relies on it to
        # undertake the data loading part.      
        self.__means_only_init__(file_path, file_alias, ref_mag, nd)
        # Calls the internal __set_cobra__() method for performing the attribute structuring task.
        self.__set_cobra__()
        
    # kistler and wake rake mode initializer.
    def __projected_and_wake_rake_init__(self, file_path, file_alias, ref_mag, nd=True):
        """Initializes an instance of the tdmsFileRoot object in "projected_and_wake_rake" reading mode.
        
        The instantiation of a tdmsFileRoot object in "projected_and_wake_rake" reading mode is to be used
        when combined Kistler and wake-rake measurements are undertaken, e.g. when performing lift and drag
        measurements together with both techniques; it builds the attribute structure corresponding to the
        wake rake surveying mode.
        
        -**parameters**::
            
        :param file_path: path of the file to be read.
        :param file_alias: string by which the data pertaining to a specific file is recognized on further charts
        or tables.
        :param ref_mag: an instantiation of the dataclass RefMagnitudes for non-dimensionalizing purposes.
        :param nd: boolean flag; if True, the non-dimensionalizing method is run on the results.
        
        """
        
        # As this initializer performs a task similar to that of the __meansOnlyInit__ method, it relies on it to
        # undertake the data loading part.
        self.__means_only_init__(file_path, file_alias, ref_mag, nd)
        # Calls the internal __project_forces__ method for performing the projecting operation on the loads.
        self.__project_forces__(self.__file__, True)
        # Calls the internal __set_kistler__() method for performing the kistler attribute structuring task.
        self.__set_kistler__()
        # Calls the internal __set_wake_rake__() method for performing the wake-rake attribute structuring task.
        self.__set_wake_rake__()
        
    # surface pressure measurements mode initializer.
    def __surface_pressure_init__(self, file_path, file_alias, ref_mag, nd=True):
        """Initializes an instance of the tdmsFileRoot object in "surface_pressure" reading mode.
        
        The instantiation of a tdmsFileRoot object in "surface_pressure" reading mode is to be used when
        performing pressure-scanner based surface measurements on bodies such as airfoils by means of
        pressure-taps or techniques alike; it builds the attribute structure corresponding to the
        pressure scanner surveying mode (similar to wake rake surveying mode).
        
        -**parameters**::
            
        :param file_path: path of the file to be read.
        :param file_alias: string by which the data pertaining to a specific file is recognized on further charts
        or tables.
        :param ref_mag: an instantiation of the dataclass RefMagnitudes for non-dimensionalizing purposes.
        :param nd: boolean flag; if True, the non-dimensionalizing method is run on the results.
        
        """
        
        # As this initializer performs a task similar to that of the __meansOnlyInit__ method, it relies on it to
        # undertake the data loading part.
        self.__means_only_init__(file_path, file_alias, ref_mag, nd)
        # Calls the internal __set_surface_pressure__() method for performing the surface-pressure attribute
        # structuring task.
        self.__set_surface_pressure__()

    #######################################################################################################
    #########################################PRIVATE INSTANCE METHODS######################################
    #######################################################################################################

    # Internal __set_root_properties__ method.
    def __set_root_properties__(self, file_path):
        """Programmatically declares runtime variables within a tdmsFileRoot object.

        The properties found at the root level of a TDMS file are programmatically declared when instantiating a
        tdmsFileRoot object. The declared name matches the one coming from the file, and the values are set as 
        "get" properties of the declared variables.

        - **properties**::

        :param file_path: path of the file to be read.

        """
        
        # An auxiliary variable (file) is declared as an instantiation of the object TdmsFile coming from the nptdms
        # package. The purpose of this variable is to provide access to the entities underlying the TDMS file to be
        # read and to allow performing the subsequent allocation of properties, both at the root level and at the
        # group/channel levels.
        # Notice that the instantiation of the TDMS file is done, exclusively, by this method; i.e. all the
        # initializers are meant to own, within their methods, either the implementation of this method or a method
        # that internally calls this one.
        self.__file__ = nptdms.TdmsFile(file_path)

        # At this level, the following "for" loop runs over the properties of the root entity of the TDMS file and
        # retrieves their names and values for the declaration and allocation of runtime variables.
        # The string replacements are meant to avoid having non-allowed characters on the declared variables, such
        # as white spaces or special characters (!"·$%&/()=?¿\).
        for prop in list(self.__file__.properties.items()):
            self.__setattr__('__' + prop[0].lower().replace(' ', '_').replace('(', '').replace(')', '') + '__', prop[1])

    # Internal __add_isolated_groups__ method.
    def __add_isolated_groups__(self, isolated_groups_name):
        """Programmatically declares an entity intended to store the groups coming from a TDMS file.

        The information found at the root level is extended by adding an entity intended to store the groups
        coming from the TDMS file being read. The declared name for this variable is a settable parameter of
        the method.

        -**parameters**, **return**, **return types**::

        :param isolated_groups_name: name of the programmatically declared variable.
        :return: __tdmsFileGroups__ object.
        :rtype: __TdmsFileGroups__

        """

        # Declaring an instantiation of the __tdmsFileGroups__ object and setting it into the "isolated_groups_name"
        # variable.
        self.__setattr__(isolated_groups_name, __TdmsFileGroups__(self.__file__, None, generate_isolated_groups=True))

    # Internal __set_standard_data__ method.
    def __set_standard_data__(self, nd=True):
        """Method that loads standard TDMS data in case the initialization has not done so.
        
        -**parameters**::
            
        :param nd: boolean flag; if True, the non-dimensionalizing method is run on the results.
        
        """

        if hasattr(self, '__groups_original__'):
            self.__setattr__('groups_original', self.__getattribute__('__groups_original__'))
            self.__delattr__('__groups_original__')
        else:
            # Conditional that assigns the attribute '__file__' to an instantiation of the nptmds.TdmsFile object in
            # order to accomplish the task of further attribute assignments.
            if hasattr(self, '__file__'):
                pass
            else:
                self.__setattr__('__file__', nptdms.TdmsFile(self.__path__))
            # Conditional that checks whether the "groups_original" attribute is already assigned to the object. If
            # so, the function does not perform further actions. Otherwise, the subsequent attribute sturcture is set.
            if hasattr(self, "groups_original"):
                return
            else:
                self.__setattr__("groups_original", __TdmsFileGroups__(self.__file__, self.__file__.groups(), generate_isolated_groups=False))

        # Conditional that, if True, calls the internal __dimensionalize__ method with the correspondent refMag object (referential magnitudes).
        if nd:
            self.__dimensionalize__(file_obj=self.__file__)

    # Internal __set_means_attributes__ method.
    def __set_mean_attributes__(self):
        """Sets the attributes necessary to deal with mean values of data.

        -**returns, return types**::

        :return: boolean indicating whether the "means_group" attribute was already set before the function call.
        :rtype: boolean.

        """

        # Conditional that assigns the attribute '__file__' to an instantiation of the nptmds.TdmsFile object in order
        # to accomplish the task of further attribute assignments.
        if hasattr(self, '__file__'):
            pass
        else:
            self.__setattr__('__file__', nptdms.TdmsFile(self.__path__))
        # Calls the internal __add_isolated_groups__ method for setting an additional internal object __tdmsFileGroups__
        # into the groups_added variable of the tdmsFileRoot object being instantiated. The groups_added variable
        # reproduces the data structure of the groups_original variable, but with derived data (means, projections,
        # Fourier transformations...).
        if hasattr(self, "groups_added"):
            pass
        else:
            self.__add_isolated_groups__("groups_added")
        # Calls the internal __add_isolated_group__ method for setting an additional internal object __tdmsFileGroup__
        # into the means_group variable of the groups_added variable of the tdmsFileRoot object being instantiated. The
        # means_group variable is intended to store the averaged values of the TDMS file data. Instead of storing the
        # data of each group separately, averaged data of the channels of all groups are stored in channels
        # corresponding to each of the variables.
        if hasattr(self.groups_added, "means_group"):
            return False
        else:
            self.groups_added.__add_isolated_group__("means_group")
            # Calls the internal __add_isolated_channels__ method for setting an additional internal object
            # __TdmsGroupChannels__ into the means_group variable; this class intends to store the channels that
            # correspond
            # to the data categories of each of the groups of the TDMS file.
            self.groups_added.means_group.__add_isolated_channels__("means_channels")

        # "True" statement return in case the structure was not set before the function call.
        return True

    # Internal __set_means__ method.
    def __set_means__(self, notif=True):
        """Rearranges group properties and channel values for their allocation on the groups_added variable.

        When employing reading modes that allocate average values of data on variables, group and channel properties
        need to be rearranged so that all the information pertaining a group (both at its level and at its channels'
        level) is included on variables within the groups_added variable.
        As an illustrating example: the positional information of a measurement (x_pos, y_pos, z_pos and angle_pos),
        which is a group level information, is the same for all the child channels of a particular group. If, say,
        the averaged channels' data is to be matched with the positional information, both the channel data and the
        positional information are required to be on a same hierarchical level. This level is provided by the
        groups_added variable, which stores positional information and averaged channel data on one-to-one
        corresponding lists.
        
        - **parameters**, **return**, **return types**::

        :param notif: boolean flag for telling whether notifications are necessary.

        """

        # Getting the current frame by inspect.
        cur_frame = inspect.currentframe()
        # Getting the outer frame to determine the caller function's name.
        cal_frame = inspect.getouterframes(cur_frame, 2)
        cal_func_name = cal_frame[1][3]   
        
        # Conditional that checks whether the caller function is the any of the means processing initializers ('__means_init__()', 
        # '__means_only_init__()').    
        if cal_func_name in ['__means_init__', '__means_only_init__']:
            # If the caller function is any containing means processing, then pass.
            pass
        else:
            if hasattr(self, 'groups_added') and hasattr(self.groups_added, 'means_group'):
                 # If the TdmsFileRoot instance already owns a 'means_group' group, then the processing is skipped.
                if notif: 
                    print('Means already processed; skipping processing.')
                return
            elif not hasattr(self.groups_added, "means_group") and hasattr(self.groups_added, "__means_group__"):
                # If the TdmsFileRoot instance owns a private '__means_group__' group, then it is necessary to perform a variable change
                # by dumping the '__means_group__' object into the public 'means_group' object.
                self.groups_added.__setattr__("means_group", self.groups_added.__means_group__)
                del self.groups_added.__means_group__

        # Declaring 'attr' variable and setting it to 'groups_original' string literal. It intends to store the name of the original group object of the TDMS file.
        attr = "groups_original"
        # Conditional for checking whether the file owns a '__groups_original__' group: if true, then the 'attr' variable is changed accordingly.
        if hasattr(self, "__groups_original__"):
            attr = "__groups_original__"

        # Conditional dependent on the call to the internal method '__set_means_attributes__()'. The conditional is executed in case
        # it is necessary to set the attribute structure.
        if self.__set_mean_attributes__():
            # A groups level "for" loop that runs over the groups on the TDMS file (excluding "Initial drift" and "Final
            # drift" groups that are not meant to provide data-analysis information).
            for group in [group for group in self.__file__.groups() if group not in ["Initial drift", "Final drift"]]:
                group = group.name.lower()
                # A group level "for" loop that runs over the properties on that group and looks for the properties
                # containing the positional information (x_pos, y_pos, z_pos and angle_pos).
                for pos_prop in [prop for prop in dir(self.__getattribute__(attr).__getattribute__(group)) if (prop.__repr__() in ["'x_pos'", "'y_pos'", "'z_pos'", "'angle_pos'"])]:
                    # Conditional that checks whether the groups_added variable already contains a child variable
                    # containing the positional information found. If so, that variable's list is appended with the
                    # newly found positional information piece; otherwise, a new variable is declared with its name
                    # matching that of the positional information, and its value being a list with a single element
                    # matching the value of the positional information.
                    if hasattr(self.groups_added.means_group.means_channels, pos_prop):
                        self.groups_added.means_group.means_channels.__getattribute__(pos_prop).append(self.__getattribute__(attr).__getattribute__(group).__getattribute__(pos_prop))
                    else:
                        self.groups_added.means_group.means_channels.__setattr__(pos_prop, [self.__getattribute__(attr).__getattribute__(group).__getattribute__(pos_prop)])
                # A channel level "for" loop that runs over the channels of the group.
                # for channel in self.__file__.group_channels(group):
                for channel in [not_dunder_property for not_dunder_property in dir(self.__getattribute__(attr).__getattribute__(group).channels) if ('__' not in not_dunder_property.__repr__())]:
                    # Conditional that checks whether the groups_added variable already contains a child variable
                    # containing the channel information found. If so, that variable's list is appended with the
                    # newly found channel information piece; otherwise, a new variable is declared with its name
                    # matching that of the channel information, and its value being a list with a single element
                    # matching the value of the averaged channel information.
                    if hasattr(self.groups_added.means_group.means_channels, channel.replace(" ", "_")):
                        self.groups_added.means_group.means_channels.__getattribute__(channel.replace(" ", "_")).append(np.average(self.__getattribute__(attr).__getattribute__(group).channels.__getattribute__(channel).data))
                    else:
                        self.groups_added.means_group.means_channels.__setattr__(channel.replace(" ", "_"), [np.average(self.__getattribute__(attr).__getattribute__(group).channels.__getattribute__(channel).data)])

    # Internal __set_wake_rake__ method.
    def __set_wake_rake__(self, wake_rake_port_ordering=tuple((np.arange(1, 19))), wake_rake_cross_axis=TDMSE.deviceAxis.y, notif=True):
        """Sets the positional information attribute structure and loads the pertaining data to it when processing wake rake data.
        
        - **parameters**::
            
        :param wake_rake_port_ordering: tuple providing ordered ports numbers from which the wake rake information is to be retrieved.
        As the wake rake owns 18 ports, by default it is considered that the ports 1-18 of the scanner are used as the terminals of the
        wake rake probes.
        :param wake_rake_cross_axis: this is a TDMSEnum custon enum class providing information of the main wind tunnel axis that has
        driven the experiment. As it is considered that, usually, the wake rake is positioned with its probes' aligned with the y-axis
        of the wind tunnel, the default value is 'deviceAxis.y'. In case any positional information is lacking, this axis is taken as the
        parameter by which the data is sorted.
        :param notif: boolean flag for telling whether notifications are necessary.
        
        """
        
        # Getting the current frame by inspect.
        cur_frame = inspect.currentframe()
        # Getting the outer frame to determine the caller function's name.
        cal_frame = inspect.getouterframes(cur_frame, 2)
        cal_func_name = cal_frame[1][3]        
        
        # Variable for determining if the group 'means_group' is deleted during the processing, or otherwise needs to be deleted.
        means_group_deleted = None
        # Conditional that checks whether the caller function is the wake rake initializer ('__wake_rake_init__()').    
        if cal_func_name == '__wake_rake_init__':
            # If the caller function is the wake rake initializer, then the means group is deleted and made private at the end of the
            # method.
            means_group_deleted = True
        else:
            # Conditional tree in case the caller function is not the wake rake initializer.
            if hasattr(self, "groups_added") and hasattr(self.groups_added, 'wake_rake_group'):
                # If the TdmsFileRoot instance already owns a 'wake_rake_group' group, then the processing is skipped.
                if notif:
                    print('Wake rake already processed; skipping processing.')
                return
            if not hasattr(self, "groups_added"):
                # If the TdmsFileRoot instance does not own a 'groups_added' group, then it is necessary to call the '__set_means__()'
                # method, and to set the 'means_group_deleted' variable to 'True'.
                self.__set_means__(notif=notif)
                means_group_deleted = True
            elif not hasattr(self.groups_added, "means_group") and not hasattr(self.groups_added, "__means_group__"):
                # If the TdmsFileRoot instance neither owns 'means_group' nor '__means_group__' groups, then it is necessary to call the
                # '__set_means__()' method, and to set the 'means_group_deleted' variable to 'True'.
                self.__set_means__(notif=notif)
                means_group_deleted = True
            elif not hasattr(self.groups_added, "means_group") and hasattr(self.groups_added, "__means_group__"):
                # If the TdmsFileRoot instance owns a private '__means_group__' group, then it is necessary to perform a variable change
                # by dumping the '__means_group__' object into the public 'means_group' object, and to set the 'means_group_deleted'
                # variable to 'True'.
                self.groups_added.__setattr__("means_group", self.groups_added.__means_group__)
                del self.groups_added.__means_group__
                means_group_deleted = True
            elif hasattr(self.groups_added, "means_group"):
                # If the TdmsFileRoot instance already owns a public 'means_group' group, then the 'means_group_deleted' varaible is set
                # to 'False'.
                means_group_deleted = False
        
        # Declaring 'eval_string' and 'runs_info' variables by calling the internal '__runs_counter__()' method; the 'runs_per_param_mode'
        # input parameter is set to 'wake_rake' accordingly, and the 'device_axis' input parameter is equated to the 'wake_rake_cross_axis'
        # input parameter of this method.
        eval_string, runs_info = self.__runs_counter__(runs_per_param_mode = TDMSE.runsPerParamMode.wake_rake, device_axis=wake_rake_cross_axis, notif=notif, toSort="wakeRake")

        # Getting the actual positional information parameter.
        actual_param = eval_string.split('__getattribute__')[-1].replace('(', '').replace(')', '').replace("'", "")
        # Getting the actual positional information parameter's index on the 'runs_info' list.
        actual_param_index = [runInfo[0] for runInfo in runs_info if runInfo[2] == actual_param][0]
        # Getting the string literal to evaluate which is the parent positional information parameter.
        parent_eval_string = ".".join(eval_string.split('.')[:-1])
        # Inline conditional for obtaining the parent eval string. If there is only one positional information parameter, then the parent
        # eval string is set to the actual eval string, as the positional information parameters are the same.
        parent_eval_string = ".".join(eval_string.split('.')[:-1]) if ([runInfo[0] for runInfo in runs_info].count(np.nan) != (len(runs_info) - 1)) else eval_string
        # Getting the parent positional information parameter.
        parent_param = parent_eval_string.split('__getattribute__')[-1].replace('(', '').replace(')', '').replace("'", "")
        # Getting the parent positional information parameter's index on the 'runs_info' list.
        parent_param_index = [runInfo[1] for runInfo in runs_info if runInfo[2] == parent_param][0]
        # Declaring the 'wake_rake_value_list' variable as an empty list. This list is intended to store the multiple wake rake values.
        wake_rake_value_list = list()
        # Declaring the 'wake_rake_temp_value_list' variable as an empty list. This list is intended to store the multiple temporal wake rake values.
        wake_rake_temp_value_list = list()

        # Loop running over the parent positional information parameter on the 'means_group' group at skips of 'parent_param_index' to ensure
        # that consecutive values of the parent positional information are retrieved.
        for i in range(0, len(self.groups_added.means_group.means_channels.__getattribute__(parent_param + '_pos')[::parent_param_index])):
            # Loop running between the current index of the parent positional information parameter and the next index of the parent positional
            # information parameter. This ensures getting all the values of the actual positional information parameter for which the parent
            # positional information parameter remains constant.
            for _ in np.arange(parent_param_index * i, parent_param_index * (i + 1)):
                # Declaring the 'wake_rake_pos' variable as an empty list. This list is intended to store the positions of the different ports
                # of the wake rake.
                wake_rake_pos = list()
                # Declaring the 'wake_rake_values' variable as an empty list. This list is intended to store the values of the different ports
                # of the wake rake.
                wake_rake_values = list()
                # Declaring the 'wake_rake_temp_values' variable as an empty list. This list is intended to store the temporal signals of the
                # different ports of the wake rake.
                wake_rake_temp_values = list()
                # Loop running over all the values of the actual positional information parameter that lie between consecutive values of the
                # parent positional information parameter.
                for actualIndex in range(0, actual_param_index):
                    # Conditional for getting the index of the actual index within the parent positional information parameter's range.
                    # This is done to ascribe the proper mean value to the 'wake_rake_values' varaible.
                    if actual_param == parent_param:
                        specific_index = actualIndex
                    else:
                        specific_index = np.arange(parent_param_index * i, parent_param_index * (i + 1))[actualIndex]
                    # Loop running over the ports of the wake rake.
                    for port in wake_rake_port_ordering:
                        # Appending the positional value of the current port to the 'wake_rake_pos' variable.
                        if self.isND():
                            wake_rake_pos.append(eval(eval_string + ".__getattribute__(" + (actual_param + "_pos").__repr__() + ")")[actualIndex] + (-wake_rake_port_ordering.index(port) + 9) * 2.505 / (1000 * self.__ref_mags__[i].length) - 1.25 / (1000 * self.__ref_mags__[i].length))
                        else:
                            wake_rake_pos.append(eval(eval_string + ".__getattribute__(" + (actual_param + "_pos").__repr__() + ")")[actualIndex] + (-wake_rake_port_ordering.index(port) + 9) * 2.505 - 1.25)
                        # Conditional to check if the current signal corresponds to the 12th port; it has been noticed that the 12th port yields
                        # abherrant data, and its value is substituted by a linear interpolation between the 11th and 13th ports.
                        if wake_rake_port_ordering.index(port) == 12:
                            # Appending wake rake values.
                            wake_rake_values.append(0.5 * (self.groups_added.means_group.means_channels.__getattribute__("port_" + str(wake_rake_port_ordering[11]) + "_signal")[specific_index] + self.groups_added.means_group.means_channels.__getattribute__("port_" + str(wake_rake_port_ordering[13]) + "_signal")[specific_index]))
                            # Conditional for correctly appending temporal wake rake signals.
                            if hasattr(self, "groups_original"):
                                wake_rake_temp_values.append(0.5 * (self.groups_original.__getattribute__("measurement" + str(specific_index + 1)).channels.__getattribute__('port_' + str(wake_rake_port_ordering[11]) + '_signal').data + self.groups_original.__getattribute__("measurement" + str(specific_index + 1)).channels.__getattribute__('port_' + str(wake_rake_port_ordering[13]) + '_signal').data))
                            else:
                                wake_rake_temp_values.append(0.5 * (self.__groups_original__.__getattribute__("measurement" + str(specific_index + 1)).channels.__getattribute__('port_' + str(wake_rake_port_ordering[11]) + '_signal').data + self.__groups_original__.__getattribute__("measurement" + str(specific_index + 1)).channels.__getattribute__('port_' + str(wake_rake_port_ordering[13]) + '_signal').data))
                        else:
                            # Appending wake rake values.
                            wake_rake_values.append(self.groups_added.means_group.means_channels.__getattribute__("port_" + str(port) + "_signal")[specific_index])
                            # Conditional for correctly appending temporal wake rake signals.
                            if hasattr(self, "groups_original"):
                                wake_rake_temp_values.append(self.groups_original.__getattribute__("measurement" + str(specific_index + 1)).channels.__getattribute__('port_' + str(port) + '_signal').data)
                            else:
                                wake_rake_temp_values.append(self.__groups_original__.__getattribute__("measurement" + str(specific_index + 1)).channels.__getattribute__('port_' + str(port) + '_signal').data)
                    # Setting the attribute corresponding to the 'wake_rake_pos' variable.
                    eval(eval_string + ".__setattr__(" + (wake_rake_cross_axis.name + "_rake_pos").__repr__() + ", wake_rake_pos)")
            # Appending the list of the retrieved wake rake values to the 'wake_rake_value_list' variable.
            wake_rake_value_list.append(wake_rake_values)
            # Appending the list of the retrieved temporal wake rake signals to the 'wake_rake_temp_value_list' variable.
            wake_rake_temp_value_list.append(wake_rake_temp_values)
        # Setting the attribute corresponding to the 'wake_rake_value_list' variable.
        eval(eval_string + ".__setattr__(" + (wake_rake_cross_axis.name + "_rake_values").__repr__() + ", wake_rake_value_list)")
        # Setting the attribute corresponding to the 'wake_rake_temp_value_list' variable.
        eval(eval_string + ".__setattr__(" + (wake_rake_cross_axis.name + "_rake_temp_values").__repr__() + ", wake_rake_temp_value_list)")

        #######################################################################################################################################
        #######################################################################################################################################
        # The following lines comprise the integration process whereby the cd values are obtained.
        #######################################################################################################################################
        #######################################################################################################################################
        # Getting the generic x-axis values for the integration process. As this generic x-axis is the same regardless of the value of the
        # parent positional information parameter, the x-axis values are computed once.
        integrate_x = [int_x for int_x in eval(eval_string + ".__getattribute__(" + (wake_rake_cross_axis.name + "_rake_pos").__repr__() + ")")]
        # Getting the list of lists containing the un-processed generic y-axis values for the integration process. This variable is termed
        # 'integration_pre_process' because its values are to be processed further before yielding a cd value.
        integration_pre_process = [eval_string + ".__getattribute__(" + (wake_rake_cross_axis.name + "_rake_values").__repr__() + ")[" + str(i) + "]" for i in list(np.arange(0, len(eval(eval_string + ".__getattribute__(" + (wake_rake_cross_axis.name + "_rake_values").__repr__() + ")"))))]
        # Declaring the 'cd' variable as an empty list.
        cd = list()
        # Declaring the 'integrates_y' variable as an empty list.
        integrates_y = list()
        # Declaring the 'stds' variable as an empty list.
        stds = list()
        # Declaring the min_max_wake_indices as an empty list.
        min_max_wake_indices = list()
        # Computing the 'parent_for_average_list'. This list simply stores the values of the parent positional information parameter.
        parent_for_average_list = eval(parent_eval_string + '.__getattribute__(' + (parent_param + '_pos').__repr__() + ')')
        # Computing the 'parent_for_average_indices' list. As currently defined, it is assumed that the default parent positional information
        # parameter is the anglewise parameter. The filtering on the below list comprehension is done so that the wake rake deficit curves
        # considered for a baseline averaging are those with a parent positional information parameter (angle) less than |5|. This is done
        # to ensure that the outer points of the curves reflect the fully unperturbed farfield flow, thus providing a sensitive baseline for
        # the averaging.
        parent_for_average_indices = [parent_for_average_list.index(x) for x in filter(lambda x: np.abs(x) <= 5, parent_for_average_list)]
        # Computing the 'actual_for_average_list'. This list simply stores the wake rake values from which to obtain the baseline averaging.
        actual_for_average_list = eval(eval_string + '.__getattribute__(' + (wake_rake_cross_axis.name + "_rake_values").__repr__() + ')')
        # Declaring the 'actual_average_list' variable as an emtpy list.
        actual_average_list = list()

        # For loop running over the indices stored in 'parent_for_average_indices'; for each of the indices, the outermost ten points of each
        # curve are appended to the 'actual_average_list' list.
        for parent_for_average_index in parent_for_average_indices:
            actual_average_list.append(np.average(actual_for_average_list[parent_for_average_index][:10]))
            actual_average_list.append(np.average(actual_for_average_list[parent_for_average_index][-10:]))
        
        # Computing the 'actual_average'; this variable simply stores the value of the baseline averaging.
        actual_average = np.average(actual_average_list)
        # The 'switch' variable either acquires a 1 or a -1 value; it is intended to post-process correctly turbulent-configuration-dependent
        # drag values.
        switch = 1
        # It is to notice that the value of the 'actual_avearage' is <0 in case turbulent-dependent measurements are performed; if so, then
        # consider the absolute value of the average for avoiding negative roots ahead, and change the value of switch to -1.
        if actual_average < 0:
            actual_average = np.abs(actual_average)
            switch = -1

        # The 'int_stopped' value is intended to halt the integration loop in case it is found that certain (but not all) values are negative,
        # which would indicate that some measurements have failed.
        int_stopped = False
        # Loop that runs over each of the un-processed deficit curves found on 'integration_pre_process'.
        for i, integrate_pre_process in enumerate(integration_pre_process):
            integrate_pre_process = eval(integrate_pre_process)
            # If any of the values in the un-processed deficit curves is negative, then enter the checking code block.
            if any([int_y < 0 for int_y in integrate_pre_process]):
                # If all the values are negative, it is an indicative that turbulent-dependent measurements are being performed; in such a case,
                # consider the absolute values of the members of 'integrate_pre_process'.
                if all([int_y < 0 for int_y in integrate_pre_process]):
                    integrate_pre_process = [np.abs(int_y) for int_y in integrate_pre_process]
                # If not all the value are negative, it means that some of the measurements (precisely the ones that show negative values) have
                # failed; halt the integration process by breaking.
                else:
                    int_stopped = True
                    break
            # Computing the 'int_list'; this list performs a variable change (from non-dimensional Pa (Cp) to deficit (u·(1-u)) and substracts
            # the averaged deficit to it).
            if self.isND():
                int_list = [np.sqrt(int_y) * (1.0 - np.sqrt(int_y)) - np.sqrt(actual_average) * (1.0 - np.sqrt(actual_average)) for int_y in integrate_pre_process]
            else:
                rho = 1.225
                u = np.average(self.groups_added.means_group.means_channels.fix_probe_signal[i: i + actual_param_index])
                average_q = 0.5*rho*u*u
                if hasattr(self.groups_added.means_group.means_channels, "ambient_density_signal"):
                    rho = np.average(self.groups_added.means_group.means_channels.ambient_density_signal[i: i + actual_param_index])
                int_list = [np.sqrt(int_y/average_q) * (1.0 - np.sqrt(int_y/average_q)) - np.sqrt(actual_average/average_q) * (1.0 - np.sqrt(actual_average/average_q)) for int_y in integrate_pre_process]            
            # Sorting the generic x-axis and pseudo-processed generic y-axis integration values according to the x-axis values.
            x, y = zip(*sorted(zip(integrate_x, int_list)))
            # Computing 'integrate_y'. This list simply stores the pseudo-processed generic y-axis integration values. Notice that the value to
            # integrate depends on the value of the 'switch' variable.
            integrate_y = [switch*int_y for int_y in y]            
            # Processing the deficit curve.
            integrate_y, std, min_max_indices = mt.deficit_curve_processor(integrate_y)            
            if any([_ < 0 for _ in integrate_y]):
                for e, _ in enumerate(integrate_y):
                    if _ < 0:
                        # integrate_y[e] *= -1
                        integrate_y[e] = 0
            # Appending the 'integrate_y' list to the 'integrates_y' list of lists. 'integrates_y' stores the processed generic y-axis
            # integration values.
            integrates_y.append(integrate_y)
            # Appending the 'std' value to the 'stds' list of standard deviations. 'stds' stores the processed standard deviations of
            # discarded data from the deficit curve.
            stds.append(std)
            # Appending the 'min_max_indices' tuple to the 'min_max_wake_indices' list of min/max indices. 'min_max_wake_indices' stores
            # the processed minimum and maximum wake indices of the filtered deficit curve.
            min_max_wake_indices.append(min_max_indices)
            # Performing the integration and appending the correspondent non-dimensional cd value (2·integration if non-dimensional;
            # 2·integration/c for dimensional) to the 'cd' list.
            if self.isND():
                cd.append(2.0 * scipy.integrate.simps(integrate_y, x=x))
            else:
                cd.append(2.0 * scipy.integrate.simps(integrate_y, x=x)/(1000 * self.__ref_mags__[0].length))
        if int_stopped:
            pass
        else:
            # Setting the attribute corresponding to the 'x' variable, which stores the generic x-axis integration values.
            eval(eval_string + ".__setattr__(" + (wake_rake_cross_axis.name + "_rake_pos").__repr__() + ", x)")
            # Setting the attribute corresponding to the 'integrates_y' variable.
            eval(eval_string + ".__setattr__(" + (wake_rake_cross_axis.name + "_rake_int_values").__repr__() + ", integrates_y)")
            # Setting the attribute corresponding to the discarded data standard deviation.
            eval(eval_string + ".__setattr__(" + (wake_rake_cross_axis.name + "_discarded_data_stds").__repr__() + ", stds)")
            # Setting the attribute corresponding to the min/max indices of filtered wake.
            eval(eval_string + ".__setattr__(" + (wake_rake_cross_axis.name + "_min_max_wake_indices").__repr__() + ", min_max_wake_indices)")
            # Setting the attribute corresponding to the 'cd' variable.
            eval(parent_eval_string + ".__setattr__(" + (parent_param + "_cd").__repr__() + ", cd)")
        
        # Restating the __means_group__ attribute if it has been previously deleted.
        if means_group_deleted:
            self.groups_added.__setattr__("__means_group__", self.groups_added.means_group)
            del self.groups_added.means_group
            
    # Internal __set_surface_pressure__ method.
    def __set_surface_pressure__(self, port_ordering=tuple((np.arange(1, 31))), port_pos = TDMSE.pressure_taps.NACA0021_elec, device_axis=TDMSE.deviceAxis.angle, notif=True):
        """Sets the positional information attribute structure and loads the pertaining data to it when processing surface pressure data.
        
        -**parameters**::
        
        :param port_ordering: tuple providing ordered ports numbers from which the surface pressure information is to be retrieved.
        By default it is considered that the ports 1-30 of the scanner are used as the terminals of the pressure taps.
        :param port_pos: list-like variable providing the positions of the pressure taps along the body at which pressure measurements are
        performed (to be retrieved from TDMSEnums.py).
        :param device_axis: this is a TDMSEnum custom enum class providing information of the main wind tunnel axis that has
        driven the experiment. As it is considered that, usually, surface pressure measurements are driven by the angular component, the
        default value is 'deviceAxis.angle'. In case any positional information is lacking, this axis is taken as the parameter by which
        the data is sorted.
        :param notif: boolean flag for telling whether notifications are necessary.        
        
        """
        
        # Getting the current frame by inspect.
        cur_frame = inspect.currentframe()
        # Getting the outer frame to determine the caller function's name.
        cal_frame = inspect.getouterframes(cur_frame, 2)
        cal_func_name = cal_frame[1][3]        
        
        # Variable for determining if the group 'means_group' is deleted during the processing, or otherwise needs to be deleted.
        means_group_deleted = None
        # Conditional that checks whether the caller function is the surface pressure initializer ('__surface_pressure_init__()').
        if cal_func_name == '__surface_pressure_init__':
            # If the caller function is the surface pressure initializers, then the means group is deleted and made private 
            # at the end of the method.
            means_group_deleted = True
        else:
            # Conditional tree in case the caller function is not the surface pressure initializer.
            if hasattr(self, "groups_added") and hasattr(self.groups_added, 'surface_pressure_group'):
                # If the TdmsFileRoot instance already owns a 'surface_pressure_group' group, then the processing is skipped.
                if notif:
                    print('Surface pressure data already processed; skipping processing.')
                return
            if not hasattr(self, "groups_added"):
                # If the TdmsFileRoot instance does not own a 'groups_added' group, then it is necessary to call the '__set_means__()'
                # method, and to set the 'means_group_deleted' variable to 'True'.
                self.__set_means__(notif=notif)
                means_group_deleted = True
            elif not hasattr(self.groups_added, "means_group") and not hasattr(self.groups_added, "__means_group__"):
                # If the TdmsFileRoot instance neither owns 'means_group' nor '__means_group__' groups, then it is necessary to call the
                # '__set_means__()' method, and to set the 'means_group_deleted' variable to 'True'.
                self.__set_means__(notif=notif)
                means_group_deleted = True
            elif not hasattr(self.groups_added, "means_group") and hasattr(self.groups_added, "__means_group__"):
                # If the TdmsFileRoot instance owns a private '__means_group__' group, then it is necessary to perform a variable change
                # by dumping the '__means_group__' object into the public 'means_group' object, and to set the 'means_group_deleted'
                # variable to 'True'.
                self.groups_added.__setattr__("means_group", self.groups_added.__means_group__)
                del self.groups_added.__means_group__
                means_group_deleted = True
            elif hasattr(self.groups_added, "means_group"):
                # If the TdmsFileRoot instance already owns a public 'means_group' group, then the 'means_group_deleted' varaible is set
                # to 'False'.
                means_group_deleted = False

        # Declaring 'eval_string' and 'runs_info' variables by calling the internal '__runs_counter__()' method; the 'runs_per_param_mode'
        # input parameter is set to 'kistler' accordingly, and the 'device_axis' input parameter is equated to the 'device_axis'
        # input parameter of this method.        
        eval_string, runs_info = self.__runs_counter__(runs_per_param_mode = TDMSE.runsPerParamMode.surface_pressure, device_axis = device_axis, notif=notif, toSort="surfacePressure")
        
        # Getting the actual positional information parameter.
        actual_param = eval_string.split('__getattribute__')[-1].replace('(', '').replace(')', '').replace("'", "")
        # Getting the string literal to evaluate which is the parent positional information parameter.
        parent_eval_string = ".".join(eval_string.split('.')[:-1])
        # Inline conditional for obtaining the parent eval string. If there is only one positional information parameter, then the parent
        # eval string is set to the actual eval string, as the positional information parameters are the same.
        parent_eval_string = ".".join(eval_string.split('.')[:-1]) if ([runInfo[0] for runInfo in runs_info].count(np.nan) != (len(runs_info) - 1)) else eval_string
        # Getting the parent positional information parameter.
        parent_param = parent_eval_string.split('__getattribute__')[-1].replace('(', '').replace(')', '').replace("'", "")
        # Getting the parent positional information parameter's index on the 'runs_info' list.
        parent_param_index = [runInfo[0] for runInfo in runs_info if runInfo[2] == parent_param][0]

        # Getting the length of the actual positional parameter.
        actual_param_length = len(self.groups_added.means_group.means_channels.__getattribute__(actual_param + '_pos'))
        
        # Conditional that runs in case there is one positional information parameter.
        if actual_param == parent_param:
            # Setting the surface pressure value attributes to a list of lists.
            eval(eval_string + ".__setattr__(" + (device_axis.name + "_surface_pressure_values").__repr__() + ", [list() for _ in range(actual_param_length)])")
            # Setting a list attribute with the positions of the pressure taps.
            eval(eval_string + ".__setattr__(" + ("pressure_taps_pos").__repr__() + ", port_pos.value)")
            # For loop running over the actual positional parameters.
            for i in range(0, actual_param_length):
                # For loop running over the mean surface presrsure values to append their values to the corresponding attributes within the 'surface_pressure_group' group.
                for port in port_ordering:
                    if port_ordering.index(port) == 12:
                        member = 0.5 * (self.groups_added.means_group.means_channels.__getattribute__("port_" + str(port - 1) + "_signal")[i] + self.groups_added.means_group.means_channels.__getattribute__("port_" + str(port + 1) + "_signal")[i])
                    else:
                        member = self.groups_added.means_group.means_channels.__getattribute__("port_" + str(port) + "_signal")[i]
                    eval(eval_string + ".__getattribute__(" + (device_axis.name + "_surface_pressure_values").__repr__() + ")[i].append(member)")
        # Conditional that runs in case there are more than one positional information parameter.
        else:
            # Loop running over the parent positional information parameter on the 'means_group' group at skips of 'parent_param_index' to ensure
            # that consecutive values of the parent positional information are retrieved.
            for i in range(0, parent_param_index):
                # The members_list variable is a list intended to store the values that correspond to measurements having the same value of the parent positional parameter.
                members_list = list()
                # For loop running over the mean surface presrsure values to append their values to the corresponding attributes within the 'surface_pressure_group' group.
                for e, port in enumerate(port_ordering):
                    if port_ordering.index(port) == 12:
                        members_list.append(0.5 * (self.groups_added.means_group.means_channels.__getattribute__("port_" + str(port - 1) + "_signal")[i*len(port_ordering) + e] + self.groups_added.means_group.means_channels.__getattribute__("port_" + str(port + 1) + "_signal")[i*len(port_ordering) + e]))
                    else:
                        members_list.append(self.groups_added.means_group.means_channels.__getattribute__("port_" + str(port) + "_signal")[i*len(port_ordering) + e])
                eval(eval_string + ".__getattribute__(" + (device_axis.name + "_surface_pressure_values").__repr__() + ")[i].append(members_list)")
        
        # Restating the __means_group__ attribute if it has been previously deleted.
        if means_group_deleted:
            self.groups_added.__setattr__("__means_group__", self.groups_added.means_group)
            del self.groups_added.means_group        
        
    # Internal __set_kistler__ method.
    def __set_kistler__(self, device_axis=TDMSE.deviceAxis.angle, notif=True):
        """Sets the positional information attribute structure and loads the pertaining data to it when processing Kistler data.
        
        - **parameters**::
            
        :param device_axis: this is a TDMSEnum custon enum class providing information of the main wind tunnel axis that has
        driven the experiment. As it is considered that, usually, the Kistler is driven by the angular component, the default value is
        'deviceAxis.angle'. In case any positional information is lacking, this axis is taken as the parameter by which the data is sorted.
        :param notif: boolean flag for telling whether notifications are necessary.
        
        """
        
        # Getting the current frame by inspect.
        cur_frame = inspect.currentframe()
        # Getting the outer frame to determine the caller function's name.
        cal_frame = inspect.getouterframes(cur_frame, 2)
        cal_func_name = cal_frame[1][3]        
        
        # Variable for determining if the group 'means_group' is deleted during the processing, or otherwise needs to be deleted.
        means_group_deleted = None
        # Conditional that checks whether the caller function is any of the Kistler initializers ('__projected_init__()',
        # 'projected_means_only_init()').    
        if cal_func_name in ['__projected_init__', '__projected_means_only_init__']:
            # If the caller function is any of the Kistler initializers, then the means group is deleted and made private at the end of the
            # method.
            means_group_deleted = True
        else:
            # Conditional tree in case the caller function is not the Kistler initializer.
            if hasattr(self, "groups_added") and hasattr(self.groups_added, 'kistler_group'):
                # If the TdmsFileRoot instance already owns a 'kistler_group' group, then the processing is skipped.
                if notif:
                    print('Kistler data already processed; skipping processing.')
                return
            if not hasattr(self, "groups_added"):
                # If the TdmsFileRoot instance does not own a 'groups_added' group, then it is necessary to call the '__set_means__()'
                # method, and to set the 'means_group_deleted' variable to 'True'.
                self.__set_means__(notif=notif)
                means_group_deleted = True
            elif not hasattr(self.groups_added, "means_group") and not hasattr(self.groups_added, "__means_group__"):
                # If the TdmsFileRoot instance neither owns 'means_group' nor '__means_group__' groups, then it is necessary to call the
                # '__set_means__()' method, and to set the 'means_group_deleted' variable to 'True'.
                self.__set_means__(notif=notif)
                means_group_deleted = True
            elif not hasattr(self.groups_added, "means_group") and hasattr(self.groups_added, "__means_group__"):
                # If the TdmsFileRoot instance owns a private '__means_group__' group, then it is necessary to perform a variable change
                # by dumping the '__means_group__' object into the public 'means_group' object, and to set the 'means_group_deleted'
                # variable to 'True'.
                self.groups_added.__setattr__("means_group", self.groups_added.__means_group__)
                del self.groups_added.__means_group__
                means_group_deleted = True
            elif hasattr(self.groups_added, "means_group"):
                # If the TdmsFileRoot instance already owns a public 'means_group' group, then the 'means_group_deleted' varaible is set
                # to 'False'.
                means_group_deleted = False

        # Declaring 'eval_string' and 'runs_info' variables by calling the internal '__runs_counter__()' method; the 'runs_per_param_mode'
        # input parameter is set to 'kistler' accordingly, and the 'device_axis' input parameter is equated to the 'device_axis'
        # input parameter of this method.        
        eval_string, runs_info = self.__runs_counter__(runs_per_param_mode = TDMSE.runsPerParamMode.kistler, device_axis = device_axis, notif=notif, toSort="kistler")       

        # Getting the actual positional information parameter.
        actual_param = eval_string.split('__getattribute__')[-1].replace('(', '').replace(')', '').replace("'", "")
        # Getting the actual positional information parameter's index on the 'runs_info' list.
        actual_param_index = [runInfo[0] for runInfo in runs_info if runInfo[2] == actual_param][0]
        # Getting the string literal to evaluate which is the parent positional information parameter.
        parent_eval_string = ".".join(eval_string.split('.')[:-1])
        # Inline conditional for obtaining the parent eval string. If there is only one positional information parameter, then the parent
        # eval string is set to the actual eval string, as the positional information parameters are the same.
        parent_eval_string = ".".join(eval_string.split('.')[:-1]) if ([runInfo[0] for runInfo in runs_info].count(np.nan) != (len(runs_info) - 1)) else eval_string
        # Getting the parent positional information parameter.
        parent_param = parent_eval_string.split('__getattribute__')[-1].replace('(', '').replace(')', '').replace("'", "")
        # Getting the parent positional information parameter's index on the 'runs_info' list.
        parent_param_index = [runInfo[0] for runInfo in runs_info if runInfo[2] == parent_param][0]

        # Declaring "drift_not_corrected_processing" boolean flag.
        drift_not_corrected_processing = False
        # Conditional for checking whether drift compensation measurements have been performed. If not, then enters a prompt dialogue with the user for
        # asking whether the processing is to be undertaken without correcting for electrical drift.
        if ("corrected_kistler_fx_signal" not in dir(self.groups_added.means_group.means_channels)):
            # Beginning prompt dialogue.
            print("Drift compensation measurements not performed: unable to correct for linear drift. Want to continue? (mind taht lift and drag forces will not be processed) y/n [y]")
            # Declaring "y_n" variable for storing user's keyboard input.
            y_n = input()
            # While loop for checking whether the user has pressed "y", "n" or "enter" keys. If it doesn't then the loop goes on until an accepted key is pressed.
            while y_n not in ["y", "n", ""]:
                print("Please enter a valid character y/n: [y]")
                y_n = input()
            # If "y" or "enter" is pressed, then the boolean flag is set to true in order to process the data without considering the unexistent corrected channels.
            if y_n in ["y", ""]:
                print("Proceeding with method without correcting for linear drift.")
                drift_not_corrected_processing = True
            # If "n" is pressed, the processing is skipped (return statement).
            else:
                print("Skipping processing.")
                return
        else:
            drift_not_corrected_processing = True

        # Declaring 'kistler_channels' as a list containing the channels pertaining Kistler data.
        kistler_channels = ["kistler_fx_signal", "kistler_fy_signal", "kistler_fz_signal", "kistler_mx_signal", "kistler_my_signal", "kistler_mz_signal", "corrected_kistler_fx_signal", "corrected_kistler_fy_signal", "corrected_kistler_fz_signal", "corrected_kistler_mx_signal", "corrected_kistler_my_signal", "corrected_kistler_mz_signal", "lift", "drag"]
        # If the processing is to be made without the corrected channels, then the "kistler_channels" variable is reset.
        if "corrected_kistler_fx_signal" not in dir(self.groups_added.means_group.means_channels) and drift_not_corrected_processing:
            kistler_channels = ["kistler_fx_signal", "kistler_fy_signal", "kistler_fz_signal", "kistler_mx_signal", "kistler_my_signal", "kistler_mz_signal"]
        # For loop running over the Kistler channels in order to set their correspondent attributes on the 'kistler_group' group.
        for channel in kistler_channels:
            # Inline conditional for setting the Kistler channel attributes to a single list (in case there is only one positional information
            # parameter) or to a list of lists (in case there are more positional information parameters).
            eval(eval_string + ".__setattr__(" + channel.__repr__() + ", [list() for _ in np.arange(0, parent_param_index)])") if [runInfo[0] for runInfo in runs_info].count(np.nan) != (len(runs_info) - 1) else eval(eval_string + ".__setattr__(" + channel.__repr__() + ", list())")

        # Conditional that runs in case there is one positional information parameter.
        if actual_param == parent_param:
            # For loop running over the Kistler channels to append their values to the corresponding attributes within the 'kistler_group' group.
            for channel in kistler_channels:
                for member in self.groups_added.means_group.means_channels.__getattribute__(channel):
                    eval(eval_string + ".__getattribute__(" + channel.__repr__() + ").append(member)")
        # Conditional that runs in case there are more than one positional information parameter.
        else:
            # Loop running over the parent positional information parameter on the 'means_group' group at skips of 'parent_param_index' to ensure
            # that consecutive values of the parent positional information are retrieved.
            for i in range(0, parent_param_index):
                # Loop running over the list of channels of the Kistler to add their mean value to the newly created 'kistler_group' group.
                for channel in kistler_channels:
                    # Loop running over each of the member of the group corresponding to the range of measurements of the current positional
                    # information parameter.          
                    for member in self.groups_added.means_group.means_channels.__getattribute__(channel)[actual_param_index*i:actual_param_index*(i + 1)]:
                        eval(eval_string + ".__getattribute__(" + channel.__repr__() + ")[i].append(member)")
                        
        # Restating the __means_group__ attribute if it has been previously deleted.
        if means_group_deleted:
            self.groups_added.__setattr__("__means_group__", self.groups_added.means_group)
            del self.groups_added.means_group
            
    # Internal __set_kistler__ method.
    def __set_cobra__(self, device_axis=TDMSE.deviceAxis.y, notif=True):
        """Sets the positional information attribute structure and loads the pertaining data to it when processing Cobra probe data.
        
        - **parameters**::
            
        :param device_axis: this is a TDMSEnum custon enum class providing information of the main wind tunnel axis that has
        driven the experiment. As it is considered that, usually, the Kistler is driven by the angular component, the default value is
        'deviceAxis.angle'. In case any positional information is lacking, this axis is taken as the parameter by which the data is sorted.
        :param notif: boolean flag for telling whether notifications are necessary.
        
        """
        
        # Getting the current frame by inspect.
        cur_frame = inspect.currentframe()
        # Getting the outer frame to determine the caller function's name.
        cal_frame = inspect.getouterframes(cur_frame, 2)
        cal_func_name = cal_frame[1][3] 
        
        # Variable for determining if the group 'means_group' is deleted during the processing, or otherwise needs to be deleted.
        means_group_deleted = None
        # Conditional that checks whether the caller function is any of the Kistler initializers ('__projected_init__()',
        # 'projected_means_only_init()').    
        if cal_func_name == '__cobra_init__':
            # If the caller function is the Cobra initializers, then the means group is deleted and made private at the end of the
            # method.
            means_group_deleted = True
        else:
            # Conditional tree in case the caller function is not the Cobra initializer.
            if hasattr(self, "groups_added") and hasattr(self.groups_added, 'cobra_group'):
                # If the TdmsFileRoot instance already owns a 'cobra_group' group, then the processing is skipped.
                if notif:
                    print('Cobra data already processed; skipping processing.')
                return
            if not hasattr(self, "groups_added"):
                # If the TdmsFileRoot instance does not own a 'groups_added' group, then it is necessary to call the '__set_means__()'
                # method, and to set the 'means_group_deleted' variable to 'True'.
                self.__set_means__(notif=notif)
                means_group_deleted = True
            elif not hasattr(self.groups_added, "means_group") and not hasattr(self.groups_added, "__means_group__"):
                # If the TdmsFileRoot instance neither owns 'means_group' nor '__means_group__' groups, then it is necessary to call the
                # '__set_means__()' method, and to set the 'means_group_deleted' variable to 'True'.
                self.__set_means__(notif=notif)
                means_group_deleted = True
            elif not hasattr(self.groups_added, "means_group") and hasattr(self.groups_added, "__means_group__"):
                # If the TdmsFileRoot instance owns a private '__means_group__' group, then it is necessary to perform a variable change
                # by dumping the '__means_group__' object into the public 'means_group' object, and to set the 'means_group_deleted'
                # variable to 'True'.
                self.groups_added.__setattr__("means_group", self.groups_added.__means_group__)
                del self.groups_added.__means_group__
                means_group_deleted = True
            elif hasattr(self.groups_added, "means_group"):
                # If the TdmsFileRoot instance already owns a public 'means_group' group, then the 'means_group_deleted' varaible is set
                # to 'False'.
                means_group_deleted = True
        
        # Declaring the "attr_original" string variable for storing the value of the name of the original group.
        attr_original = "groups_original"
        # Conditional for obtaining the existent original group object, be it named either "__groups_original__" or "groups_original".
        if hasattr(self, "__groups_original__"):
            attr_original = "__groups_original__"
        
        U_avg_list = list()
        ucin_avg_list = list()
        vcin_avg_list = list()
        wcin_avg_list = list()
        Ucin_avg_list = list()
        uvwcin_avg_list = list()
        iuu_avg_list = list()
        ivv_avg_list = list()
        iww_avg_list = list()
        iuvw_avg_list = list()
        # For loop for setting the fluctuating, turbulent kinetic energy and intensity temporal signals in the original group.        
        for group in sorted([self.__getattribute__(attr_original).__getattribute__(group) for group in dir(self.__getattribute__(attr_original)) if "measurement" in group], key=lambda x: int(x.group_name.split("Measurement")[-1])):
            # Adding isolated channels to the Cobra group if necessary.
            if not hasattr(group.channels, "cobra_ufluc_comp_signal"):
                group.channels.__add_isolated_channel__("cobra_ufluc_comp_signal")
                group.channels.__add_isolated_channel__("cobra_ucin_comp_signal")
                group.channels.__add_isolated_channel__("cobra_vfluc_comp_signal")
                group.channels.__add_isolated_channel__("cobra_vcin_comp_signal")
                group.channels.__add_isolated_channel__("cobra_wfluc_comp_signal")
                group.channels.__add_isolated_channel__("cobra_wcin_comp_signal")
                group.channels.__add_isolated_channel__("cobra_U_comp_signal")
                group.channels.__add_isolated_channel__("cobra_Ufluc_comp_signal")
                group.channels.__add_isolated_channel__("cobra_Ucin_comp_signal")
                group.channels.__add_isolated_channel__("cobra_uvwcin_comp_signal")
                group.channels.__add_isolated_channel__("cobra_iuu_signal")
                group.channels.__add_isolated_channel__("cobra_ivv_signal")
                group.channels.__add_isolated_channel__("cobra_iww_signal")
                group.channels.__add_isolated_channel__("cobra_iuvw_signal")
            if hasattr(group.channels, "cobra_velocity_signal"):
                # Temporal velocity vector.
                vel = group.channels.cobra_velocity_signal.data
                # Average velocity vector.             
                vel_avg = np.average(vel)
                # Temporal u component.
                u_comp = group.channels.cobra_u_comp_signal.data
                # Average u component.
                u_comp_avg = np.average(u_comp)
                # Temporal v component.
                v_comp = group.channels.cobra_v_comp_signal.data
                # Average v component.
                v_comp_avg = np.average(v_comp)
                # Temporal w component.
                w_comp = group.channels.cobra_w_comp_signal.data
                # Average w compnent.
                w_comp_avg = np.average(w_comp)
                # Temporal U component.
                U_comp = np.sqrt(u_comp**2 + v_comp**2 + w_comp**2)
                # Average U component.
                U_comp_avg = np.average(U_comp)
                # Computing u fluctuations and setting them to "cobra_ufluc_comp_signal" variable of the original group.
                ufluc = u_comp - u_comp_avg                
                group.channels.cobra_ufluc_comp_signal.data = ufluc
                # Computing u turbulent kinetic energies and setting them to "cobra_ucin_comp_signal" variable of the original group.
                ucin = ufluc**2
                group.channels.cobra_ucin_comp_signal.data = ucin
                ucin_avg_list.append(np.average(ucin))
                # Computing v fluctuations and setting them to "cobra_vfluc_comp_signal" variable of the original group.
                vfluc = v_comp - v_comp_avg
                group.channels.cobra_vfluc_comp_signal.data = vfluc
                # Computing v turbulent kinetic energies and setting them to "cobra_ucin_comp_signal" variable of the original group.
                vcin = vfluc**2                
                group.channels.cobra_vcin_comp_signal.data = vcin
                vcin_avg_list.append(np.average(vcin))
                # Computing w fluctuations and setting them to "cobra_wfluc_comp_signal" variable of the original group.
                wfluc = w_comp - w_comp_avg                
                group.channels.cobra_wfluc_comp_signal.data = wfluc
                # Computing w turbulent kinetic energies and setting them to "cobra_ucin_comp_signal" variable of the original group.
                wcin = wfluc**2                
                group.channels.cobra_wcin_comp_signal.data = wcin
                wcin_avg_list.append(np.average(wcin))
                # Setting U component to the "cobra_U_comp_signal" variable of the original group and appending its mean value to
                # 'U_avg_list'.                
                group.channels.cobra_U_comp_signal.data = U_comp
                U_avg_list.append(np.average(U_comp))
                # Computing U flutuations and setting them to "cobra_Ufluc_comp_signal" variable of the original group.
                Ufluc = U_comp - U_comp_avg                
                group.channels.cobra_Ufluc_comp_signal.data = Ufluc
                # Computing U turbulent kinetic energies and setting them to "cobra_Ucin_comp_signal" variable of the original group.
                Ucin = ucin + vcin + wcin                
                group.channels.cobra_Ucin_comp_signal.data = Ucin
                Ucin_avg_list.append(np.average(Ucin))
                # Computing overall turbulent kinetic energies and setting them to "cobra_uvwcin_comp_signal" variable of the original group.
                uvwcin = (ucin + vcin + wcin)/3                
                group.channels.cobra_uvwcin_comp_signal.data = uvwcin
                uvwcin_avg_list.append(np.average(uvwcin))
                # Computing iuu intensity and setting it to "cobra_iuu_signal" variable of the original group.            
                # iuu = 100*np.sqrt(np.mean(u_comp**2))/vel_avg
                iuu = 100*np.sqrt(np.mean(ufluc**2))/vel_avg
                iuu_avg_list.append(iuu)
                group.channels.cobra_iuu_signal.data = iuu*np.ones(len(u_comp))
                # Computing ivv intensity and setting it to "cobra_ivv_signal" variable of the original group.
                ivv = 100*np.sqrt(np.mean(vfluc**2))/vel_avg
                ivv_avg_list.append(ivv)
                group.channels.cobra_ivv_signal.data = ivv*np.ones(len(v_comp))
                # Computing iww intensity and setting it to "cobra_iww_signal" variable of the original group.                
                iww = 100*np.sqrt(np.mean(wfluc**2))/vel_avg
                iww_avg_list.append(iww)
                group.channels.cobra_iww_signal.data = iww*np.ones(len(w_comp))
                # Computing iuvw (intensity vector) and setting it to "cobra_iuvw_signal" variable of the original group.                
                iuvw = np.sqrt((1/3)*(np.average(ufluc**2) + np.average(vfluc**2) + np.average(wfluc**2)))/vel_avg
                iuvw_avg_list.append(iuvw)
                group.channels.cobra_iuvw_signal.data = iuvw*np.ones(len(w_comp))
        
        # Declaring the "attr_means" string variable for storing the value of the name of the means group.
        attr_means = str()
        # Conditional for obtaining the existent means group object, be it named either "__means_group__" or "means_group".
        if means_group_deleted:
            attr_means = "means_group"
        else:
            attr_means = "__means_group__"
        # Adding mean velocity and turbulent intensity means to their correspondent mean groups.
        self.groups_added.__getattribute__(attr_means).means_channels.__setattr__("cobra_ucin_signal", ucin_avg_list)
        self.groups_added.__getattribute__(attr_means).means_channels.__setattr__("cobra_vcin_signal", vcin_avg_list)
        self.groups_added.__getattribute__(attr_means).means_channels.__setattr__("cobra_wcin_signal", wcin_avg_list)
        self.groups_added.__getattribute__(attr_means).means_channels.__setattr__("cobra_U_signal", U_avg_list)
        self.groups_added.__getattribute__(attr_means).means_channels.__setattr__("cobra_Ucin_signal", Ucin_avg_list)
        self.groups_added.__getattribute__(attr_means).means_channels.__setattr__("cobra_uvwcin_signal", uvwcin_avg_list)
        self.groups_added.__getattribute__(attr_means).means_channels.__setattr__("cobra_iuu_signal", iuu_avg_list)
        self.groups_added.__getattribute__(attr_means).means_channels.__setattr__("cobra_ivv_signal", ivv_avg_list)
        self.groups_added.__getattribute__(attr_means).means_channels.__setattr__("cobra_iww_signal", iww_avg_list)
        self.groups_added.__getattribute__(attr_means).means_channels.__setattr__("cobra_iuvw_signal", iuvw_avg_list)
        
        # Declaring 'eval_string' and 'runs_info' variables by calling the internal '__runs_counter__()' method; the 'runs_per_param_mode'
        # input parameter is set to 'kistler' accordingly, and the 'device_axis' input parameter is equated to the 'device_axis'
        # input parameter of this method.        
        eval_string, runs_info = self.__runs_counter__(runs_per_param_mode = TDMSE.runsPerParamMode.cobra, device_axis = device_axis, notif=notif, toSort="cobra")      
        
        # Getting the actual positional information parameter.
        actual_param = eval_string.split('__getattribute__')[-1].replace('(', '').replace(')', '').replace("'", "")
        # Getting the actual positional information parameter's index on the 'runs_info' list.
        actual_param_index = [runInfo[0] for runInfo in runs_info if runInfo[2] == actual_param][0]
        # Inline conditional for obtaining the parent eval string. If there is only one positional information parameter, then the parent
        # eval string is set to the actual eval string, as the positional information parameters are the same.
        parent_eval_string = ".".join(eval_string.split('.')[:-1]) if ([runInfo[0] for runInfo in runs_info].count(np.nan) != (len(runs_info) - 1)) else eval_string
        # Getting the parent positional information parameter.
        parent_param = parent_eval_string.split('__getattribute__')[-1].replace('(', '').replace(')', '').replace("'", "")
        # Getting the parent positional information parameter's index on the 'runs_info' list.
        parent_param_index = [runInfo[0] for runInfo in runs_info if runInfo[2] == parent_param][0] if len([runInfo[0] for runInfo in runs_info if runInfo[2] == parent_param]) != 0 else 1
        
        # Declaring 'cobra_channels' as a list containing the channels pertaining Cobra data.
        cobra_channels = ["cobra_velocity_signal", "cobra_u_comp_signal", "cobra_v_comp_signal", "cobra_w_comp_signal", "cobra_ucin_signal", "cobra_vcin_signal", "cobra_wcin_signal", "cobra_uvwcin_signal", "cobra_iuu_signal", "cobra_ivv_signal", "cobra_iww_signal", "cobra_iuvw_signal", "cobra_pitch_angle_signal", "cobra_yaw_angle_signal"]
        # For loop running over the Cobra channels in order to set their correspondent attributes on the 'cobra_group' group.
        for channel in cobra_channels:
            # Inline conditional for setting the Cobra channel attributes to a single list (in case there is only one positional information
            # parameter) or to a list of lists (in case there are more positional information parameters).
            eval(eval_string + ".__setattr__(" + channel.__repr__() + ", [list() for _ in np.arange(0, parent_param_index)])") if [runInfo[0] for runInfo in runs_info].count(np.nan) != (len(runs_info) - 1) else eval(eval_string + ".__setattr__(" + channel.__repr__() + ", list())")

        # Conditional that runs in case there is one positional information parameter.
        if actual_param == parent_param:
            # For loop running over the Cobra channels to append their values to the corresponding attributes within the 'cobra_group' group.
            for channel in cobra_channels:
                for member in self.groups_added.means_group.means_channels.__getattribute__(channel):
                    eval(eval_string + ".__getattribute__(" + channel.__repr__() + ").append(member)")
        # Conditional that runs in case there are more than one positional information parameter.
        else:
            # Loop running over the parent positional information parameter on the 'means_group' group at skips of 'parent_param_index' to ensure
            # that consecutive values of the parent positional information are retrieved.
            for i in range(0, parent_param_index):
                # Loop running over the list of channels of the cobra probe to add their mean value to the newly created 'cobra_group' group.
                for channel in cobra_channels:
                    # Loop running over each of the member of the group corresponding to the range of measurements of the current positional
                    # information parameter.
                    for member in self.groups_added.means_group.means_channels.__getattribute__(channel)[actual_param_index*i:actual_param_index*(i + 1)]:
                        eval(eval_string + ".__getattribute__(" + channel.__repr__() + ")[i].append(member)")
        
        # Restating the __means_group__ attribute if it has been previously deleted.
        if means_group_deleted:
            self.groups_added.__setattr__("__means_group__", self.groups_added.means_group)
            del self.groups_added.means_group

    # Internal __runs_counter__ method.
    def __runs_counter__(self, runs_per_param_mode, device_axis=TDMSE.deviceAxis.y, notif=True, toSort="kistler"):
        """Checks whether the experimental runs are sequentially stacked regarding the positional information.
        
        When a multiparametric (from a positional standpoint) experimental test is run, the information
        pertaining the order of the positional arguments acquires relevance when it comes to sort the data.
        This method performs such a task by taking in parameters that provide relevant information on the
        positional variation that the experiment to be processed owns.
        
        - **parameters**, **return**, **return types**::

        :param runs_per_param_mode: this is a TDMSEnums custon enum class providing information of
        what type of experiment is to be processed. Attributes such as the name of the group to be added
        are determined by this value.
        :param device_axis: this is a TDMSEnum custon enum calss providing information of the main
        wind tunnel axis that has driven the experiment. In case any positional information is lacking,
        this axis is taken as the parameter by which to sort the data.
        :param notif: boolean flag for telling whether notifications are necessary.
        :param toSort: string literal for performing a filtering on the measurements' group.
        Default is 'kistler', but it may well be 'wakeRake' or 'cobra'.
        :return: eval_string variable, runs_info variable.
        :rtype: string, list        
        
        """

        # Declaring a list to store the information regarding the experimental runs.  
        runs_info = list()

        # Loop that runs over the different axes defined in the custom enum TDMSEnums.deviceAxis and appending the positional
        # information to the runs_info variable accordingly.
        for pos_info in [axis.name for axis in TDMSE.deviceAxis]:
            runs_info.append(
                # True value of in-line conditional for a 4-member tuple: (Member 1, Member 2, Member 3, Member 4).
                (
                    # Member 1: number of different runs for a given positional information; obtained by computing the length of the list of
                    # the current positional information divided by the minimum repeated counts on that list.
                    len(self.groups_added.means_group.means_channels.__getattribute__(pos_info + "_pos")) // min([self.groups_added.means_group.means_channels.__getattribute__(pos_info + "_pos").count(run_value) for run_value in self.groups_added.means_group.means_channels.__getattribute__(pos_info + "_pos")]),
                    # len(self.groups_added.means_group.means_channels.__getattribute__(pos_info + "_pos")) // [i != self.groups_added.means_group.means_channels.__getattribute__(pos_info + "_pos")[0] for i in self.groups_added.means_group.means_channels.__getattribute__(pos_info + "_pos")].index(True),
                    # Member 2: index on which the positional information changes its value; obtained by a list comprehension statement
                    # considering the first 'True' value of an inequality operator between an iterating index 'i' and the first value of
                    # the list of the current conditional information.
                    [i != self.groups_added.means_group.means_channels.__getattribute__(pos_info + "_pos")[0] for i in self.groups_added.means_group.means_channels.__getattribute__(pos_info + "_pos")].index(True),
                    # Member 3: name of the positional information.
                    pos_info,
                    # Member 4: consecutive runs per positional information, referring to the member #2 of the list.
                    "consecutive runs per " + pos_info
                )
                # Condition: checks whether the minimum count of repeated values for a given positional information list does not coincide
                # with that list's length. If it does, it means that all the values are the same; i.e. the positional information parameter
                # does not change among runs, and hence it can be neglected.
                if min([self.groups_added.means_group.means_channels.__getattribute__(pos_info + "_pos").count(run_value) for run_value in self.groups_added.means_group.means_channels.__getattribute__(pos_info + "_pos")]) != len(self.groups_added.means_group.means_channels.__getattribute__(pos_info + "_pos"))
                # False value of in-line conditional for a 4-member tuple: (Member 1, Member 2, Member 3, Member 4).
                else
                # Member 1: np.NaN if the current positional information parameter does not coincide with the input device axis name;
                # 1 otherwise.
                (np.NaN if (pos_info != device_axis.name) else len(self.groups_added.means_group.means_channels.__getattribute__(pos_info + "_pos")),
                # Member 2: np.NaN if the current positional information parameter does not coincide with the input device axis name;
                # 1 otherwise.
                 np.NaN if (pos_info != device_axis.name) else 1,
                 # Member 3: empty if the current positional information parameter does not coincide with the input device axis name;
                 # current positional information parameter's name otherwise.
                 "" if (pos_info != device_axis.name) else pos_info,
                 # Member 4: empty if the current positional information parameter does not coincide with the input device axis name;
                 # 'consecutive runs per "pos_info"' otherwise, where "pos_info" is the current positional information parameter's name.
                 "" if (pos_info != device_axis.name) else "consecutive runs per" + pos_info)
            )

        # Sorting the 'runs_info' variable by using its lists' second values as keys (number of consecutive runs); the sorting is done
        # in reverse order, meaning that the runs info is sorted beginning from the positional information parameter that owns the smallest
        # number of consecutive runs.
        sorted_runs_info = sorted([run_info for run_info in runs_info if not np.isnan(run_info[1])], key=lambda x: x[1], reverse=True)
        
        ####################################################################################################################################
        # The following code block aims at allowing the post-processing of TDMS files corresponding to combined protocolized tests, such
        # as Kistler/wake-rake tests for obtaining lift/drag curves in a single wind tunnel run. The logic is the following:
        #       1) assume that a joint lift/drag test with Kistler/wake-rake methods is to be run. For each angular position of the
        # airfoil, a single Kistler measurement is taken, whereas several (up to 7) measurements with the wake-rake are necessary in
        # order to obtain a complete momentum-deficit curve. This sets forth the need of decoupling the Kistler measurements with the
        # wake-rake ones within the same test.
        #       2) the code block that follows performs a filtering on the measurements' group on the basis of an input parameter,
        # 'toSort'. The value of such a parameter tells which measurements are to be kept for obtaining the correspondent positional
        # information that the current method, '__runs_counter__()', provides.
        #       3) modifying the information provided by the '__runs_counter__()' method is necessary due to the internal logic of
        # the overall code. If this is not done and, as mentioned, a single Kistler measurement is taken against 7 wake-rake measurements
        # for each angular configuration, then the code would automatically detect that the positional information for, say, a set of angles
        # of attack spanning the range [0, 20] is (21, 7) -> (angle, y), which is true for the wake-rake case, but it is not for the Kistler
        # one. This latter should be (21) -> (angle), and there resides the necessity of modifying the code. Such is the intention of the
        # developed code.
        ####################################################################################################################################        
        # The 'st' variable intends to store the string literal that is to be used for checking which are the groups that own specific
        # measurement channels, such as Kistler, wake-rake or Cobra measurements.
        st = str()
        # Conditional tree for assigning a value to 'st' attending to the input parameter 'toSort'.
        # 'kistler' if 'toSort'=='kistler', 'port' if 'toSort'=='wakeRake', 'cobra if 'toSort'=='cobra' and 'port' if 'toSort'=='surfacePressure'.
        if toSort == "kistler":            
            st = "kistler"
        elif toSort == "wakeRake":
            st="port"
        elif toSort == "cobra":
            st = "cobra"
        elif toSort == "surfacePressure":
            st = "port"
        # The 'attr' variable stores the string literal corresponding to either '__groups_original__' or 'groups_original', depending on which
        # of them actually is an attribute of the file.
        attr = "groups_original"
        if hasattr(self, "__groups_original__"):
            attr = "__groups_original__"
        # The 'indices' variables stores the indices resulting from the measurement filtering performed on the basis of 'st'.
        indices = sorted([int(meas.split("measurement")[1]) - 1 for meas in dir(self.__getattribute__(attr)) if "__" not in meas if "initial_drift" not in meas if "final_drift" not in meas if any([st in _ for _ in dir(self.__getattribute__(attr).__getattribute__(meas).channels)])]) if st != "" else list()
        # Conditional for checking whether the length of the filtered measurements equals the totality of the measurements; if not,
        # then proceed to performing the modification in the 'sorted_runs_info' and 'runs_info' variables; otherwise, set the 'indices'
        # variable back to a list.        
        if len(indices) != np.nanprod([runs_per_param[0] for runs_per_param in sorted_runs_info]):
            # Obtaining the index for which the modification is to be performed, which is the one owning the length of the 'indices'
            # variable (the length of the filtered measurements).
            index = [_[0]==len(indices) for _ in sorted_runs_info].index(True)
            # Obtaining the indices of the 'runs_info' variable that whose first argument does not coincide with the length of the
            # filtered measurements. This is done for obtaining the information of the remaining positional arguments, which are
            # the ones to be modified.
            runs_info_indices = np.where(np.array([_ == __ for _ in runs_info for __ in sorted_runs_info if __[0] != len(indices)]) == True)
            # Modifying the 'sorted_runs_info' variable at the index of the filtered variable's positional argument. Setting it to
            # ("", 1, "", ""), where "" represents that the original value is kept.
            sorted_runs_info[index] = (sorted_runs_info[index][0], 1, sorted_runs_info[index][2], sorted_runs_info[index][3])
            # 'for' loop for modifying the 'runs_info' variable at the positional arguments other than the one corresponding to the
            # filtered measurements. Modifying them to (np.NaN, np.NaN, "", ""), where "" represents an empty string literal.
            for runs_info_index in runs_info_indices[0]:
                runs_info[runs_info_index] = (np.NaN, np.NaN, "", "")
            # 'for' loop for modifying the 'sorted_runs_info' variable at the positional arguments other than the one corresponding
            # to the filtered measurements. Modifying them to (np.NaN, np.NaN, "", ""), where "" represents an empty string literal.
            for i in range(index + 1, len(sorted_runs_info)):
                sorted_runs_info[i] = (np.NaN, np.NaN, "", "")
        else:
            indices = list()
        # Recomputing the 'sorted_runs_info' variable for getting rid of the NaN values.
        sorted_runs_info = sorted([run_info for run_info in sorted_runs_info if not np.isnan(run_info[1])], key=lambda x: x[1], reverse=True)
        
        # Checking whether the length of the 'sorted_runs_info' parameter, after tripping the NaN values, is equal to 1, and if the first two
        # values of the tuple of that unique member are also 1. If so, it means that no positional information parameters have been varied
        # during the test, and the 'sorted_runs_info' member is changed accordingly, setting a new tuple with its first member corresponding
        # to the number of measurements carried out.
        if (len(sorted_runs_info) == 1) and (sorted_runs_info[0][0] == 1) and (sorted_runs_info[0][1] == 1):
            sorted_runs_info[0] = (len(self.groups_added.means_group.means_channels.__getattribute__(pos_info + "_pos")), 1, sorted_runs_info[0][2], sorted_runs_info[0][3])

        # Getting the values to be assigned (member #1 according to the previous tuple).
        to_assign_values = [runInfo[0] for runInfo in sorted_runs_info]
        # Getting the consecutive values to be assigned (member #2 according to the previous tuple).
        to_assign_con_values = [run_info[1] for run_info in sorted_runs_info]
        # Getting the positional information parameter names to be assigned (member #3 according to the previous tuple).
        to_assign_param_names = [run_info[2] for run_info in sorted_runs_info]

        # Getting the name of the group to be added to the current object instance; the name is taken from the input 'runs_per_param_mode'
        # parameter and adding the term 'group' to it.
        added_group = runs_per_param_mode.name + "_group"
        # Creating an instance of the '__RunsPerParam__' internal object that represents the group to be added.
        runs_per_param_group = __RunsPerParam__(added_group)
        # Setting the attribute to the 'groups_added' attribute of the current object instance.
        self.groups_added.__setattr__(added_group, runs_per_param_group)
        # Deleting the attribute for avoiding duplicates.
        self.groups_added.__getattribute__(added_group).__delattr__(added_group)

        # Conditional that checks whether the product (avoiding NaN values) of the counted runs coincide with the length of any of the lists
        # of positional information parameters; if so , it proceeds to the ordered instantiation of positional information objects.
        length = len(self.groups_added.means_group.means_channels.z_pos) if len(indices) == 0 else len(indices)
        if np.nanprod([runs_per_param[0] for runs_per_param in sorted_runs_info]) == length:
            if notif:
                print(colored("Sequential runs: check correct --> Ordering: (" + ",".join([str(sorted_run_info[2]) for sorted_run_info in sorted_runs_info]) + ") --> Numbering: (" + ",".join([str(sorted_run_info[0]) for sorted_run_info in sorted_runs_info]) + ")", color="green", attrs=["bold"]))
            # Loop that runs over the names of the parameters to be assigned.
            for param_name in to_assign_param_names:
                # If the name of the parameter to be assigned is the first on the list, then the instantiation is done directly on the
                # 'added_group' attribute previously set on the 'groups_added' attribute of the current object instance.
                if to_assign_param_names.index(param_name) == 0:
                    # Conditional for checking whether the length of the filtered variables equals 0; if it does not, then perform the
                    # value picking in accordance to the 'indices' variable.
                    if len(indices) == 0:
                        runs_per_param = __RunsPerParam__(param_name + "_pos", self.groups_added.means_group.means_channels.__getattribute__(param_name + "_pos")[::to_assign_con_values[to_assign_param_names.index(param_name)]])
                    # Otherwise, perform the value picking in a standard fashion.
                    else:
                        runs_per_param = __RunsPerParam__(param_name + "_pos", np.array(self.groups_added.means_group.means_channels.__getattribute__(param_name + "_pos"))[indices])
                    self.groups_added.__getattribute__(added_group).__setattr__(param_name, runs_per_param)
                    if len(to_assign_param_names) == 1:
                        eval_string = "self.groups_added.__getattribute__(" + added_group.__repr__() + ").__getattribute__(" + param_name.__repr__() + ")" 
                # If the name of the parameter to be assigned is not the first on the list, then it is necessary to check the position in
                # which it has to be set; the 'eval_string' variable is a string-type variable that is modified by recursive
                # '__getattribute__()' substrings seed by all the previous positional information parameters to be taken into account.
                else:
                    eval_string = "self.groups_added.__getattribute__(" + added_group.__repr__() + ")."
                    for previous_param_name in to_assign_param_names[:to_assign_param_names.index(param_name)]:
                        eval_string += "__getattribute__(" + previous_param_name.__repr__() + ")."
                    # Conditional for checking whether the length of the filtered variables equals 0; if it does not, then perform the
                    # value picking in accordance to the 'indices' variable.                    
                    if len(indices) == 0 and len(to_assign_values) != 0:
                        eval(eval_string + "__setattr__(" + param_name.__repr__() + ", __RunsPerParam__(" + (param_name + "_pos").__repr__() + ", self.groups_added.means_group.means_channels.__getattribute__(" + (param_name + "_pos").__repr__() + ")[:to_assign_values[to_assign_param_names.index(" + param_name.__repr__() + ")]:to_assign_con_values[to_assign_param_names.index(" + param_name.__repr__() + ")]]))")
                    # Otherwise, perform the value picking in a standard fashion.
                    else:
                        eval(eval_string + "__setattr__(" + param_name.__repr__() + ", __RunsPerParam__(" + (param_name + "_pos").__repr__() + ", np.array(self.groups_added.means_group.means_channels.__getattribute__(" + (param_name + "_pos").__repr__() + "))[indices]))")
                    # Conditional that checks whether the list of parameter names to be assigned has been exhausted; if so, it means that
                    # the positional information parameter to be set needs to be introduced at this innermost level.
                    if to_assign_param_names.index(param_name) == len(to_assign_param_names) - 1:
                        eval_string = eval_string.split("__setattr__")[0] + "__getattribute__(" + param_name.__repr__() + ")"
            # Return statement.
            return eval_string, runs_info        
        # If the above conditional is "False", then a message is displayed telling that the TDMS file being processed is not liable to be
        # ordered coherently by levelled runs.
        else:
            if notif:
                print(colored("Check not correct. File not liable to be ordered coherently by levelled runs.", color="red", attrs=["bold"]))
            # Return statement.
            return
        
    # Internal __get_position_hierarchy__ method.
    def __get_position_hierarchy__(self, departing_group: TDMSE.addedGroups):
        '''Obtains the positional hierarchy information considering the input group as the apex of the hierarchical pyramid.
        
        - **parameters**, **return**, **return types**::
        
        :param departing_group: a TDMSE.addedGroups instance addressing the group among the ones present in the 'groups_added'
        group of groups from which to depart the hierarchical classification.
        :return: eval_string variable, runs_info present_pos_params variable.
        :rtype: string, list
        
        '''
        
        # Declaring the 'eval_string' variable and assigning it to a string suited for computing an eval statement for getting the input group.
        eval_string = 'self.groups_added.' + departing_group.name
        # Declaring the 'pos_params' variable and assigning it to a list containing the positional information names defined in the custom enum
        # TDMSEnums.deviceAxis.
        pos_params = [axis.name for axis in TDMSE.deviceAxis]
        # Declaring the 'present_pos_params' variable and assigning it to an empty list. This variable is intended to store the positional
        # information parameters that show up when performing the hierarchichal classification.
        present_pos_params = list()
        # Declaring the 'boolean_flag' parameter and assigning it to a 'True' value. This variable is intended to act as a stopping variable
        # of the while loop that will run over the attribute hierarchy of the group.
        boolean_flag = True        
        # While loop running over the attribute hierarchy of the group looking for positional information.
        while boolean_flag:
            # The 'evaled_param' variable is assigned to the evaled computation of the 'eval_string' variable.
            evaled_param = eval(eval_string)
            # The 'bool_pos_params' variable is assigned to a list-comprehension-computed list containing boolean information about the current
            # 'evaled_param' paremter owning any of the positional information parameters.
            bool_pos_params = [hasattr(evaled_param, pos_param) for pos_param in pos_params]
            # The 'boolean_flag' parameter is equated to the boolean value of the sum of the 'bool_pos_params' list.
            boolean_flag = bool(np.sum(bool_pos_params))
            # If 'boolean_flag' is 'True', it means that a positional information parameter is present, and it is necessary to travel a level
            # deeper in search of additional positional information paramters.
            if boolean_flag:
                # As such, the last positional information parameter found is appended to the 'present_pos_params' variable.
                present_pos_params.append(pos_params[bool_pos_params.index(True)])
                # The 'eval_string' variable is modified by adding the corresponding '__getattribute__' method pertaining to the last positional
                # information parameter found.
                eval_string += '.__getattribute__(' + pos_params[bool_pos_params.index(True)].__repr__() + ')'
                
        # Return statement.
        return eval_string, present_pos_params

    # Internal __project_forces__ method.
    def __project_forces__(self, file_obj, means_only=False):
        """Projects force-related data on the TDMS file into wind tunnel axes.

        The projection of force-related data on the TDMS file into wind tunnel axes is done according to the positional
        information on each of the measurement groups. Mind that the load balance is assumed to be aligned with wind
        tunnel axes for a 0º configuration, with x- and y-axis being flow-wise and traverse directions, respectively.
        Clockwise angles are assumed positive.

        - **parameters**::

        :param file_obj: a nptdms.TdmsFile object containing non_drift_group/channel information to be projected.
        :param means_only: boolean flag signaling whether averaged values are meant to be considered exclusively.

        """

        # Declaring 'attr_original' variable and setting it to 'groups_original' string literal. It intends to store the name of the original
        # group object of the TDMS file.
        attr_original = "groups_original"
        # Conditional for checking whether the file owns a '__groups_original__' group: if true, then the 'attr' variable is changed accordingly.
        if hasattr(self, "__groups_original__"):
            attr_original = "__groups_original__"

        # Declaring 'attr_means' variable and setting it to 'means_group' string literal. It intends to store the name of the means
        # group object of the TDMS file.
        attr_means = "means_group"
        # Conditional for checking whether the file owns a '__means_group__' group: if true, then the 'attr' variable is changed accordingly.
        if hasattr(self.groups_added, "__means_group__"):
            attr_means = "__means_group__"

        # A groups level "for" loop that runs over the groups on the TDMS file (excluding "Initial drift" and "Final
        # drift" groups that are not meant to provide data-analysis information).
        for non_drift_group in [group for group in file_obj.groups() if group not in ["Initial drift", "Final drift"]]:
            # A group_obj looping variable is declared that instantiates a group object of the fileObj variable.
            group_obj = file_obj.object(non_drift_group)
            # A channel_list looping variable is declared that contains a list of the channels on the group
            # "non_drift_group" of the file_Obj variable.
            channel_list = [channel.channel for channel in file_obj.group_channels(non_drift_group)]
            # Conditional for checking whether the non_drift_group "non_drift_group" contains, in its channels,
            # drift-compensated load measurements.
            if all(elem in channel_list for elem in ["Kistler signal time", "Corrected Kistler Fx signal"]):
                # Getting the angle at which the measurement was taken from the positional information, and converting
                # it to radian units.
                angle = group_obj.properties.get("Angle pos") * np.pi / 180
                # Computing drag and lift projections according to:
                # d = Fx·cos(a) - Fy·sin(a)
                # length = Fx·sin(a) + Fy·cos(a)
                drag = np.cos(angle) * file_obj.object(non_drift_group, "Corrected Kistler Fx signal").data - np.sin(angle) * file_obj.object(non_drift_group, "Corrected Kistler Fy signal").data
                lift = np.sin(angle) * file_obj.object(non_drift_group, "Corrected Kistler Fx signal").data + np.cos(angle) * file_obj.object(non_drift_group, "Corrected Kistler Fy signal").data
                # Conditional that checks on the boolean flag; if false (temporal information is meant to be damped),
                # additional attributes of "drag" and "lift" are declared at the channel level of the groups_original
                # variable, and their values set to the computed values of drag and lift.
                if not means_only:
                    self.__getattribute__(attr_original).__getattribute__(non_drift_group.lower()).__setattr__("drag", drag)
                    self.__getattribute__(attr_original).__getattribute__(non_drift_group.lower()).__setattr__("lift", lift)
                # Conditional that checks whether the groups_added variable already contains a child variable
                # containing the projected information. If so, that variable's list is appended with the newly
                # found projected information piece; otherwise, a new variable is declared with its name matching
                # that of the projected information, and its value being a list with a single element matching the
                # value of the projected information.
                if hasattr(self.groups_added.__getattribute__(attr_means).means_channels, "drag"):
                    self.groups_added.__getattribute__(attr_means).means_channels.drag.append(np.average(drag))
                    self.groups_added.__getattribute__(attr_means).means_channels.lift.append(np.average(lift))
                else:
                    self.groups_added.__getattribute__(attr_means).means_channels.__setattr__("drag", [np.average(drag)])
                    self.groups_added.__getattribute__(attr_means).means_channels.__setattr__("lift", [np.average(lift)])

    # Internal __dimensionalize__ method.
    def __dimensionalize__(self, file_obj, mode='non', notif=True):
        """Non-dimensionalizes data coming from the TDMS file.

        - **parameters**::

        :param file_obj: a nptdms.TdmsFile object containing group/channel information to be projected.
        :param mode: string literal 'non/re' indicating whether a non-dimensionalization or a re-dimensionalization is to be performed.
        Default is 'non'.
        :param notif: boolean flag for telling whether notifications are necessary.
        Default is True.        

        """

        # Assertion statement for assessing that the 'mode' parameter is correctly passed.
        assert mode in ['non', 're'], """Please provide a valid dimensionalizing mode: either 'non' for non-dimensionalization,
        or 're' for re-dimensionalization."""

        # Declaring 'attr' variable and setting it to 'groups_original' string literal. It intends to store the name of the original group object of the TDMS file.
        attr = "groups_original"
        # Conditional for checking whether the file owns a '__groups_original__' group: if true, then the 'attr' variable is changed accordingly.
        if hasattr(self, "__groups_original__"):
            attr = "__groups_original__"

        # Declaring the 'groups_added_list' variable which is meant to store the list of added groups in order to non/re-dimensionalize them
        # after the original group is processed.
        groups_added_list = list()
        # Conditional for checking whether there are added groups. In case there are, the labels of the ones added by TDMS processing methods are acquired.
        if hasattr(self, "groups_added"):
            groups_added_list = [added_group for added_group in dir(self.groups_added) if "__" not in added_group or "__means_group__" in added_group]
            # For loop for deleting the groups so that they are newly processed further on.
            for added_group in groups_added_list:
                delattr(self.groups_added, added_group)

        # Conditonal tree for checking whether any processing is to be undertaken, depending on the passed 'mode' and the currentframe
        # non-dimensionalization state (self.__nd_d__) of the TDMS file.
        # If 'mode' is 'non' and the file is already non-dimensionalized, then skip process:
        if mode=='non' and self.__nd_d__==True:
            if notif:
                print("File data already in non-dimensional format. Skipping process.")
            pass
        # If 'mode' is 're' and the file is already dimensionalized, then skip process:
        elif mode=='re' and self.__nd_d__==False:
            if notif:
                print("File data already in dimensional format. Skipping process.")
            pass
        # If 'mode' is 'non' and the file is not non-dimensionalized, then perform non-dimensionalization:
        elif mode=='non' and self.__nd_d__==False:
            # Checking whether drift measurements have been performed (group names 'initial drift' and 'final drift' exist). If so, then the 'drift_index' parameter
            # is assigned the index 1; 0 otherwise.
            drift_index = 1 if "initial drift" in [meas_group.name.lower().replace("''", 'prima').replace("/", "").replace("''", '"').replace("'", "").replace("prima", "'").split('"')[0] for meas_group in file_obj.groups()] else 0
            # Getting the number of measurements performed (accounting for potential drift measruements).
            measurements = len([meas for meas in dir(self.__getattribute__(attr)) if "__" not in meas])
            # For loop running over the object meas_groups found on the TDMS file.
            for meas_group in file_obj.groups():
                # Declaring 'meas_group_meas_index' variable and assigning to it a null value.
                meas_group_index = 0
                # Conditional tree for getting the measurement index corresponding to its cardinal order on the overall set of measurements (including drifts).
                if "measurement" in meas_group.name.lower().replace("''", 'prima').replace("/", "").replace("''", '"').replace("'", "").replace("prima", "'").split('"')[0]:
                    meas_group_index = int(meas_group.name.lower().replace("''", 'prima').replace("/", "").replace("''", '"').replace("'", "").replace("prima", "'").split('"')[0].split("measurement")[1]) - 1 if not drift_index else int(meas_group.name.lower().replace("''", 'prima').replace("/", "").replace("''", '"').replace("'", "").replace("prima", "'").split('"')[0].split("measurement")[1])
                elif "final" in meas_group.name.lower().replace("''", 'prima').replace("/", "").replace("''", '"').replace("'", "").replace("prima", "'").split('"')[0]:
                    meas_group_index = measurements - drift_index
                # Getting the name of the meas_group.
                meas_group_name_list = meas_group.name.lower().replace("''", 'prima').replace("/", "").replace("''", '"').replace("'", "").replace("prima", "'").split('"')                
                # Conditional for checking whether a name has been found (first condition) and whether that name is not the root name (empty string).
                if (len(meas_group_name_list) == 1) and (meas_group_name_list[0] != ""):
                    # Conditional for checking whether pressure, rh and temperature data are available; if True, then the referential magnitudes are set accordingly.
                    if hasattr(self.__getattribute__(attr).__getattribute__(meas_group_name_list[0].replace(" ", "_")).__getattribute__("channels"), "barometric_pressure_signal"):
                        self.__ref_mags__[meas_group_index].p = np.average(self.__getattribute__(attr).__getattribute__(meas_group_name_list[0].replace(" ", "_")).__getattribute__("channels").__getattribute__("barometric_pressure_signal").data)
                        self.__ref_mags__[meas_group_index].rh = np.average(self.__getattribute__(attr).__getattribute__(meas_group_name_list[0].replace(" ", "_")).__getattribute__("channels").__getattribute__("relative_humidity_signal").data)
                        self.__ref_mags__[meas_group_index].temp = np.average(self.__getattribute__(attr).__getattribute__(meas_group_name_list[0].replace(" ", "_")).__getattribute__("channels").__getattribute__("ambient_temperature_signal").data) + 273.15
                    # Conditional for checking whether velocity data is available; if True, then the referential magnitude is set accordingly.
                    if hasattr(self.__getattribute__(attr).__getattribute__(meas_group_name_list[0].replace(" ", "_")).__getattribute__("channels"), "fix_probe_signal"):
                        self.__ref_mags__[meas_group_index].u = np.average(self.__getattribute__(attr).__getattribute__(meas_group_name_list[0].replace(" ", "_")).__getattribute__("channels").__getattribute__("fix_probe_signal").data)
                    # Non-dimensionalizing x-axis related positional information.                    
                    self.__getattribute__(attr).__getattribute__(meas_group_name_list[0].replace(" ", "_")).x_pos /= (1000 * self.__ref_mags__[meas_group_index].length)
                    # Non-dimensionalizing y-axis related positional information.
                    self.__getattribute__(attr).__getattribute__(meas_group_name_list[0].replace(" ", "_")).y_pos /= (1000 * self.__ref_mags__[meas_group_index].length)
                    # Non-dimensionalizing z-axis related positional information.
                    self.__getattribute__(attr).__getattribute__(meas_group_name_list[0].replace(" ", "_")).z_pos /= (1000 * self.__ref_mags__[meas_group_index].length)
                # Conditional for checking whether non-dimensionalization operations are needed on the current meas_group's properties. The "unit_string"
                # flag adresses magnitudes that are to be non-dimensionalized.                           
                for channel in file_obj.groups()[meas_group_index].channels():
                    # Getting channel's name.
                    channame = channel.name
                    # Getting channel's properties.
                    channprops = channel.properties
                    # Conditional for determining the magnitude that is to be non-dimensionalized.
                    if "unit_string" in channprops:
                        # Non-dimensionalizing pressure-related (Pa) magnitudes, in case the referential magnitude is meant to be q (dynamic pressure) and q is not 0.
                        if ("Barometric" not in channame) and (channprops["unit_string"] == "Pa") and (self.__ref_mags__[meas_group_index].q):                        
                            self.__getattribute__(attr).__getattribute__(meas_group_name_list[0].replace(" ", "_")).channels.__getattribute__(channame.lower().replace(" ", "_")).data /= self.__ref_mags__[meas_group_index].q
                        # Non-dimensionalizing pressure-related (Pa) magnitudes, in case the referential magnitude is meant to be p.
                        elif ("Barometric" in channame) and (channprops["unit_string"] == "Pa"):
                            self.__getattribute__(attr).__getattribute__(meas_group_name_list[0].replace(" ", "_")).channels.__getattribute__(channame.lower().replace(" ", "_")).data /= self.__ref_mags__[meas_group_index].p
                        # Non-dimensionalizing rh-related (%) magnitudes.
                        elif (channprops["unit_string"] == "%"):
                            self.__getattribute__(attr).__getattribute__(meas_group_name_list[0].replace(" ", "_")).channels.__getattribute__(channame.lower().replace(" ", "_")).data /= self.__ref_mags__[meas_group_index].rh
                        # Non-dimensionalizing density-related (kg/m^3) magnitudes.
                        elif (channprops["unit_string"] == "kg/m^3"):
                            self.__getattribute__(attr).__getattribute__(meas_group_name_list[0].replace(" ", "_")).channels.__getattribute__(channame.lower().replace(" ", "_")).data /= self.__ref_mags__[meas_group_index].rho
                        # Non-dimensionalizing viscosity-related (kg/ms) magnitudes.
                        elif (channprops["unit_string"] == "kg/ms"):
                            self.__getattribute__(attr).__getattribute__(meas_group_name_list[0].replace(" ", "_")).channels.__getattribute__(channame.lower().replace(" ", "_")).data /= self.__ref_mags__[meas_group_index].mu
                        # Non-dimensionalizing force-related (N) magnitudes, in case the referential magnitude q (dynamic pressure) is not 0.
                        elif channprops["unit_string"] == "N" and self.__ref_mags__[meas_group_index].q:
                            self.__getattribute__(attr).__getattribute__(meas_group_name_list[0].replace(" ", "_")).channels.__getattribute__(channame.lower().replace(" ", "_")).data /= (self.__ref_mags__[meas_group_index].q * self.__ref_mags__[meas_group_index].length * self.__ref_mags__[meas_group_index].span)
                        # Non-dimensionalizing torque-related (N·m) magnitudes, in case the referential magnitude q (dynamic pressure) is not 0.
                        elif channprops["unit_string"] == "Nm" and self.__ref_mags__[meas_group_index].q:
                            self.__getattribute__(attr).__getattribute__(meas_group_name_list[0].replace(" ", "_")).channels.__getattribute__(channame.lower().replace(" ", "_")).data /= (self.__ref_mags__[meas_group_index].q * (self.__ref_mags__[meas_group_index].length ** 2.0) * self.__ref_mags__[meas_group_index].span)
                        # Non dimensionalizing velocity-related (m/s) magnitudes.
                        elif channprops["unit_string"] == "m/s":
                            self.__getattribute__(attr).__getattribute__(meas_group_name_list[0].replace(" ", "_")).channels.__getattribute__(channame.lower().replace(" ", "_")).data /= self.__ref_mags__[meas_group_index].u
                        # Non-dimensionalizing temperature-related (ºC) magnitudes, by converting the values to ºK (+273) and dividing by the
                        # corresponding referential magnitude later.
                        elif channprops["unit_string"] == "ºC":
                            self.__getattribute__(attr).__getattribute__(meas_group_name_list[0].replace(" ", "_")).channels.__getattribute__(channame.lower().replace(" ", "_")).data += 273
                            self.__getattribute__(attr).__getattribute__(meas_group_name_list[0].replace(" ", "_")).channels.__getattribute__(channame.lower().replace(" ", "_")).data /= self.__ref_mags__[meas_group_index].temp
                        else:
                            pass
            # Setting the self.__nd_d__ parameter according to the performed process.
            self.__nd_d__ = True
        # If 'mode' is 're' and the file is non-dimensionalized, then perform dimensionalization:
        elif mode=='re' and self.__nd_d__==True:
            # Checking whether drift measurements have been performed (group names 'initial drift' and 'final drift' exist). If so, then the 'drift_index' parameter
            # is assigned the index 1; 0 otherwise.
            drift_index = 1 if "initial drift" in [item[0].lower().replace("''", 'prima').replace("/", "").replace("''", '"').replace("'", "").replace("prima", "'").split('"')[0] for item in file_obj.objects.items()] else 0
            # Getting the number of measurements performed (accounting for potential drift measruements).
            measurements = len([meas for meas in dir(self.__getattribute__(attr)) if "__" not in meas])
            # For loop running over the each object items found on the TDMS file.
            for item in file_obj.objects.items():
                # Declaring 'item_meas_index' variable and assigning to it a null value.
                item_meas_index = 0
                # Conditional tree for getting the measurement index corresponding to its cardinal order on the overall set of measurements (including drifts).
                if "measurement" in item[0].lower().replace("''", 'prima').replace("/", "").replace("''", '"').replace("'", "").replace("prima", "'").split('"')[0]:
                    item_meas_index = int(item[0].lower().replace("''", 'prima').replace("/", "").replace("''", '"').replace("'", "").replace("prima", "'").split('"')[0].split("measurement")[1]) - 1 if not drift_index else int(item[0].lower().replace("''", 'prima').replace("/", "").replace("''", '"').replace("'", "").replace("prima", "'").split('"')[0].split("measurement")[1])
                elif "final" in item[0].lower().replace("''", 'prima').replace("/", "").replace("''", '"').replace("'", "").replace("prima", "'").split('"')[0]:
                    item_meas_index = measurements - drift_index
                # Getting the name of the item.
                item_name_list = item[0].lower().replace("''", 'prima').replace("/", "").replace("''", '"').replace("'", "").replace("prima", "'").split('"')
                # Conditional for checking whether a name has been found (first condition) and whether that name is not the root name (empty string).
                if (len(item_name_list) == 1) and (item_name_list[0] != ""):
                    # Non-dimensionalizing x-axis related positional information.
                    self.__getattribute__(attr).__getattribute__(item_name_list[0].replace(" ", "_")).x_pos *= (1000 * self.__ref_mags__[item_meas_index].length)
                    # Non-dimensionalizing y-axis related positional information.
                    self.__getattribute__(attr).__getattribute__(item_name_list[0].replace(" ", "_")).y_pos *= (1000 * self.__ref_mags__[item_meas_index].length)
                    # Non-dimensionalizing z-axis related positional information.
                    self.__getattribute__(attr).__getattribute__(item_name_list[0].replace(" ", "_")).z_pos *= (1000 * self.__ref_mags__[item_meas_index].length)
                # Conditional for checking whether non-dimensionalization operations are needed on the current item's properties. The "unit_string"
                # flag adresses magnitudes that are to be non-dimensionalized.
                if "unit_string" in item[1].properties:
                    # Non-dimensionalizing pressure-related (Pa) magnitudes, in case the referential magnitude is meant to be q (dynamic pressure) and q is not 0.
                    if ("Barometric" not in item[0]) and (item[1].property("unit_string") == "Pa") and (self.__ref_mags__[item_meas_index].q):
                        self.__getattribute__(attr).__getattribute__(item_name_list[0].replace(" ", "_")).channels.__getattribute__(item_name_list[1].replace(" ", "_")).data *= self.__ref_mags__[item_meas_index].q
                    # Non-dimensionalizing pressure-related (Pa) magnitudes, in case the referential magnitude is meant to be p.
                    elif ("Barometric" in item[0]) and (item[1].property("unit_string") == "Pa"):
                        self.__getattribute__(attr).__getattribute__(item_name_list[0].replace(" ", "_")).channels.__getattribute__(item_name_list[1].replace(" ", "_")).data *= self.__ref_mags__[item_meas_index].p
                    # Non-dimensionalizing rh-related (%) magnitudes.
                    elif (item[1].property("unit_string") == "%"):
                        self.__getattribute__(attr).__getattribute__(item_name_list[0].replace(" ", "_")).channels.__getattribute__(item_name_list[1].replace(" ", "_")).data *= self.__ref_mags__[item_meas_index].rh
                    # Non-dimensionalizing density-related (kg/m^3) magnitudes.
                    elif (item[1].property("unit_string") == "kg/m^3"):
                        self.__getattribute__(attr).__getattribute__(item_name_list[0].replace(" ", "_")).channels.__getattribute__(item_name_list[1].replace(" ", "_")).data *= self.__ref_mags__[item_meas_index].rho
                    # Non-dimensionalizing viscosity-related (kg/ms) magnitudes.
                    elif (item[1].property("unit_string") == "kg/ms"):
                        self.__getattribute__(attr).__getattribute__(item_name_list[0].replace(" ", "_")).channels.__getattribute__(item_name_list[1].replace(" ", "_")).data *= self.__ref_mags__[item_meas_index].mu
                    # Non-dimensionalizing force-related (N) magnitudes, in case the referential magnitude q (dynamic pressure) is not 0.
                    elif item[1].property("unit_string") == "N" and self.__ref_mags__[item_meas_index].q:
                        self.__getattribute__(attr).__getattribute__(item_name_list[0].replace(" ", "_")).channels.__getattribute__(item_name_list[1].replace(" ", "_")).data *= (self.__ref_mags__[item_meas_index].q * self.__ref_mags__[item_meas_index].length * self.__ref_mags__[item_meas_index].span)
                    # Non-dimensionalizing torque-related (N·m) magnitudes, in case the referential magnitude q (dynamic pressure) is not 0.
                    elif item[1].property("unit_string") == "Nm" and self.__ref_mags__[item_meas_index].q:
                        self.__getattribute__(attr).__getattribute__(item_name_list[0].replace(" ", "_")).channels.__getattribute__(item_name_list[1].replace(" ", "_")).data *= (self.__ref_mags__[item_meas_index].q * (self.__ref_mags__[item_meas_index].length ** 2.0) * self.__ref_mags__[item_meas_index].span)
                    # Non dimensionalizing velocity-related (m/s) magnitudes.
                    elif item[1].property("unit_string") == "m/s":
                        self.__getattribute__(attr).__getattribute__(item_name_list[0].replace(" ", "_")).channels.__getattribute__(item_name_list[1].replace(" ", "_")).data *= self.__ref_mags__[item_meas_index].u
                    # Non-dimensionalizing temperature-related (ºC) magnitudes, by converting the values to ºK (+273) and dividing by the
                    # corresponding referential magnitude later.
                    elif item[1].property("unit_string") == "ºC":
                        self.__getattribute__(attr).__getattribute__(item_name_list[0].replace(" ", "_")).channels.__getattribute__(item_name_list[1].replace(" ", "_")).data *= self.__ref_mags__[item_meas_index].temp
                        self.__getattribute__(attr).__getattribute__(item_name_list[0].replace(" ", "_")).channels.__getattribute__(item_name_list[1].replace(" ", "_")).data -= 273
                    else:
                        pass
            # Setting the self.__nd_d__ parameter according to the performed process.
            self.__nd_d__ = False

        # For loop for reprocessing previously added groups for non/re-dimensionalizing.
        for added_group in groups_added_list:
            # Conditional for checking whether data averages are to be processed.
            if "means" in added_group:
                self.process_means(notif=False)
            # Conditional for checking whether a kistler data structure is to be processed.
            elif "kistler" in added_group:
                if hasattr(self, "groups_added"):
                    if not hasattr(self.groups_added, "__means_group__") or not hasattr(self.groups_added, "means_group"):
                        self.process_means(notif=False)
                self.__project_forces__(self.__file__)
                self.__set_kistler__(notif=False)
            # Conditional for checking whether a wake-rake data structure is to be processed.
            elif "wake" in added_group:
                self.process_wake_rake(notif=False)
            # Conditional for checking whether a cobra data structure is to be processed.
            elif "cobra" in added_group:                
                self.process_cobra(notif=False)
            # Conditional for checking whether a surface pressure data structure is to be processed.
            elif "surface" in added_group:
                self.process_surface_pressure(notif=False)

        # Return statement.
        return
    
    # Internal __corrector__ method.
    def __corrector__(self, mode='kistler'):
        '''Corrections due to wall interferences.
        
        Performs lift, drag, pitch momentum, velocity and angle-of-attack corrections due to wall
        interferences, according to the formulae of 2011Selig.
        
        - **parameters**::

        :param mode: a string literal, either 'kistler' or 'wakeRake', indicating which drag coefficient
        data is to be employed for correcting the curves (the corrections are Cd-value-dependent.)
        
        '''
        
        ## Assertions.
        # Asserting that the input 'mode' parameter is either 'kistler' or 'wakeRake'.
        assert mode in ['kistler', 'wakeRake'], "Please provide a valid 'mode' input argument ('kistler' or 'wakeRake')."
        # Asserting that the current TDMS file owns a 'groups_added' group; otherwise, letting the user
        # know that he/she must run the 'process_kistler()' method.
        assert hasattr(self, "groups_added"), "Please process Kistler data, self.process_kistler()."
        # Asserting that the current TDMS file owns a 'kistler_group' group; otherwise, letting the user
        # know that he/she must run the 'process_kistler()' method.
        assert hasattr(self.groups_added, "kistler_group"), "Please process Kistler data, self.process_kistler()."
        # Conditional for checking whether the input parameter has the 'wakeRake' value.
        if mode=='wakeRake':
            # Asserting that the current TDMS file owns a 'wake_rake_group' group; otherwise, letting the
            # user know that he/she must run the 'process_wake_rake()' method.
            assert hasattr(self.groups_added, "wake_rake_group"), "Please process wake-rake group, self.process_wake_rake()."
        
        # Checking whether the TDMS file is non-dimensionalized; if it isn't, then non-dimensionalize.
        if not self.isND():
            self.dimensionalize(mode='non', notif=False)
        
        # The 'attr' variable intends to store the name that addresses the variable storing the original
        # groups within the TDMS file; either "groups_original" or "__groups_original__", depending on
        # the fact that the TDMS file has been 'standard-processed' or not.
        attr = "groups_original" if hasattr(self, "groups_original") else "__groups_original__"
        # The 'means_attr' variable intends to store the name that addresses the variable storing the
        # means group within the TDMS file; either "means_group" or "__means_group__", depending on the
        # fact that the TDMS file has been 'means-processed' or not.
        means_attr = "means_group" if hasattr(self.groups_added, "means_group") else "__means_group__"
        # Obtaining the angles of attack.
        angles = np.array(self.groups_added.kistler_group.angle.angle_pos)
        # Obtaining the lift coefficient.
        cl = np.array(self.groups_added.kistler_group.angle.lift)
        # Obtaining the pitch momentum coefficient.
        cm = np.array(self.groups_added.kistler_group.angle.corrected_kistler_mx_signal)
        # Declaring the 'cd' variable for storing the drag coefficient.
        cd = np.array(list())
        # The 'filt_index' variable intends to store the jumps by which the velocity variable list is to
        # be filtered; it acquires the unit value in case all the measurements stored within the TDMS file
        # own Kistler-related channels, but this is not the case when Kistler and wake-rake measurements
        # have been performed jointly; in such a case, it is usual to have more wake-rake-related channels
        # than Kistler-related ones, due to the fact that it is necessary to perform a number of measurements
        # per angular configuration in order to obtain a complete momentum-deficit curve. This makes that
        # the measurements owning a Cl value are less than the ones owning a velocity value, so that the
        # velocities need to be filtered in accordance to the number of Cls within the TDMS file.
        filt_index = len([_ for _ in dir(self.__getattribute__(attr)) if "measurement" in _])//len(cl)
        # Filtering the velocity list according to the just computed 'filt_index' value.
        vels = np.array(self.groups_added.__getattribute__(means_attr).means_channels.fix_probe_signal[::filt_index])
        # Conditional tree for obtaining the drag coefficient, either from the drag coefficients of the
        # Kistler group or of the wake-rake group, depending on the value of the input parameter 'mode'.
        if mode=="kistler":
            cd = np.abs(np.array(self.groups_added.kistler_group.angle.drag))
        elif mode=="wakeRake":
            cd = np.array(self.groups_added.wake_rake_group.angle.angle_cd)
        
        # Declaring 'l' variable for storing average value of characteristic length. Setting it to the value of its homologous referential magnitude.
        l = self.__ref_mags__[0].length
        # Declaring 's' variable for storing average value of characteristic span. Setting it to the value of its homologous referential magnitude.
        s = self.__ref_mags__[0].span
        # Declaring 't' variable for storing average value of characteristic thickness. Setting it to the value of its homologous referential magnitude.
        t = self.__ref_mags__[0].thick        
        ###
        ### Correction parameter computation.
        ###
        ## Geometric parameters.
        # Wind tunnel height value.
        H = 0.902
        # Wind tunnel span value.
        S = 0.75
        # C area constant, equaling the product of wind tunnel height and span.
        C = H * S
        # K1 constant, equaling 0.52 for an airfoil spanning the tunnel's height.
        K1 = 0.52
        # V volume constant, equaling 0.7 times the product of the characteristic dimensions.
        V = 0.7 * t * l * s
        ## Flow-dependent parameters.
        # Solid blockage.
        esb = (K1 * V) / C**(1.5)
        # Wake blockage.
        wb = cd * (l / (2 * H))
        # Blockage.
        eb = esb + wb
        # Sigma.
        sigma = (np.pi**2 / 48) * (l / H)**2
        ## Performing corrections.
        # Correcting angle of attack.
        anglescorr = angles + ((57.3*sigma)/(2*np.pi))*(cl + 4*cm)
        # Correcting lift coefficient.
        clcorr = cl*(1 - sigma)/(1 + eb)**2
        # Correcting drag coefficient.
        cdcorr = cd*(1 - esb)/(1 + eb)**2
        # Correcting pitch momentum coefficient.
        cmcorr = (cm + cl*sigma*(1 - sigma)/4)/(1 + eb)**2
        # Correcting velocity.
        velscorr = vels*(1 + eb)
        
        ## Storing the computed corrected values in their correpondent variables.
        # The following conditional tree checks whether the variables for the corrected
        # velocities already exist (the correction has been previously applied) or not.
        # If they do, then merely the values of those variables is set to the newly computed
        # values. Otherwise, they are created and set to the computed values.
        if hasattr(self.groups_added.kistler_group.angle, "anglescorr"):
            # Setting corrected angle values to 'anglescorr' variable.
            self.groups_added.kistler_group.angle.anglescorr = anglescorr
            # Setting corrected lift values to 'clcorr' variable.
            self.groups_added.kistler_group.angle.clcorr = clcorr
            # If 'mode'=='kistler' and the Kistler group contains the variable 'cdcorr',
            # then setting corrected drag values to 'cdcorr' variable.
            if mode=='kistler' and hasattr(self.groups_added.kistler_group.angle, "cdcorr"):
                self.groups_added.kistler_group.angle.cdcorr = cdcorr
            # If the Kistler group does not own a 'cdcorr' variable, then creating it and
            # setting its value to the corrected drag values.
            elif mode=='kistler' and not hasattr(self.groups_added.kistler_group.angle, "cdcorr"):
                self.groups_added.kistler_group.angle.__setattr__("cdcorr", cdcorr)
            # If 'mode'=='kistler' and the wake-rake group contains the variable 'cdcorr',
            # then setting corrected drag values to 'cdcorr' variable.
            elif mode=='wakeRake' and hasattr(self.groups_added.wake_rake_group.angle, "cdcorr"):
                self.groups_added.wake_rake_group.angle.cdcorr = cdcorr
            # If the wake-rake group does not own a 'cdcorr' variable, then creating it and
            # setting its value to the corrected drag values.                
            else:
                self.groups_added.wake_rake_group.angle.__setattr__("cdcorr", cdcorr)
            # Setting corrected pitch momentum values to 'cmcorr' variable.
            self.groups_added.kistler_group.angle.cmcorr = cmcorr
            # Setting corrected velocity values to 'velscorr' variable.
            self.groups_added.kistler_group.angle.velscorr = velscorr
            # Setting the employed 'mode' parameter value to 'correction_basis' variable, in
            # order to register what mode has been employed in the correction.
            self.groups_added.kistler_group.angle.correction_basis = mode
        else:
            # Creating the 'anglescorr' variable and setting its value to the corrected angle values.
            self.groups_added.kistler_group.angle.__setattr__("anglescorr", anglescorr)
            # Creating the 'clcorr' variable and setting its value to the corrected lift values.
            self.groups_added.kistler_group.angle.__setattr__("clcorr", clcorr)
            # If 'mode'=='kistler', creating the 'cdcorr' variable and setting its value to the
            # corrected drag values.
            if mode=='kistler':
                self.groups_added.kistler_group.angle.__setattr__("cdcorr", cdcorr)
            # If 'mode'=='wakeRake' and the wake-rake group already has a 'cdcorr' variable, then
            # setting its value to the corrected drag values.
            elif mode=='wakeRake' and hasattr(self.groups_added.wake_rake_group.angle, "cdcorr"):
                self.groups_added.wake_rake_group.angle.cdcorr = cdcorr
            # If 'mode'=='wakeRake' and the wake-rake group does not have a 'cdcorr' variable, then
            # creating the 'cdcorr' variable and setting its value to the corrected drag values.
            else:
                self.groups_added.wake_rake_group.angle.__setattr__("cdcorr", cdcorr)
            # Creating the 'cmcorr' variable and setting its value to the corrected pitch momentum values.
            self.groups_added.kistler_group.angle.__setattr__("cmcorr", cmcorr)
            # Creating the 'velscorr' variable and setting its value to the corrected velocity values.
            self.groups_added.kistler_group.angle.__setattr__("velscorr", velscorr)
            # Creating the 'correction_basis' variable and setting its value to the 'mode' parameter
            # value, in order to reister what mode has been employed in the correction.
            self.groups_added.kistler_group.angle.__setattr__("correction_basis", mode)
            
        # Return statement.
        return

    # Internal __set_uncertainty_intervals__ method.
    def __set_uncertainty_intervals__(self, NVFx=5, NVFy=5, NVFz=5, delta_l=1.4e-3, delta_s=0.7e-3, delta_p=7.5e-2):
        """Performs an uncertainty analysis on the data of the TDMS file.

        -**parameters**

        :param NVFx: N/V scale of the piezoelectric's x-axis.
        :param NVFy: N/V scale of the piezoelectric's y-axis.
        :param NVFz: N/V scale of the piezoelectric's z-axis.
        :param delta_l: uncertainty interval corresponding to the characteristic length.
        :param delta_s: uncertainty interval corresponding to the characteristic span.
        :param delta_p: uncertainty interval corresponding to pressure scanner measurements.

        """
        
        # Conditional for checking whether current data is already uncertainty-analysed. If so, then skip process.
        if self.__unc_d__:
            print("File data already uncertainty-analysed. Skipping process.")
            return

        # Conditional for checking whether the TDMS file is means-averaged, and performing such a process if it is not.
        if hasattr(self, "groups_added"):
            if not hasattr(self.groups_added, "means_group") or not hasattr(self.groups_added, "__means_group__"):
                self.process_means(notif=False)
        else:
            self.process_means(notif=False)

        # Dimensionalizing data if necessary.
        re_dimensionalized = False
        if self.isND():
            re_dimensionalized = True
            self.dimensionalize(mode='re', notif=False)           
        else:
            self.dimensionalize(notif=False)
            self.dimensionalize(mode='re', notif=False)

        # Declaring 'attr' variable and setting it to 'groups_original' string literal. It intends to store the name of the original group object of the TDMS file.
        attr = "groups_original"
        # Conditional for checking whether the file owns a '__groups_original__' group: if true, then the 'attr' variable is changed accordingly.
        if hasattr(self, "__groups_original__"):
            attr = "__groups_original__"

        # Getting measurements in sorted format according to the measurement number.
        measurements = sorted([meas for meas in dir(self.__getattribute__(attr)) if "measurement" in meas], key=lambda i: int(i.split("measurement")[1]))
        # 'driftspresent' boolean that tells whether drift measurements have been performed, in which case 'initial_drift' and 'final_drift' groups exist.
        # This forces to renumber the indices for accessing the referential magnitudes stored in the TDMS file object.
        driftspresent = False
        # Conditional for setting the 'driftspresent¡ variable to "True" in case drift measurements have been performed.
        if len(measurements) != len([meas for meas in dir(self.__getattribute__(attr)) if meas[:2] != "__"]):
            driftspresent = True
        # Declaring 'unc_list' variable and assigning to it an empty list. This variable intends to store the 'uncertainty_intervals' object instantiations
        # for each of the measurements.
        unc_list = list()
        # for loop running over the measurements in the sorted order given by the 'pot_meas' list. It performs an uncertainty analysis on each of the
        # measurements.        
        for measurement in measurements:
            i = int(measurement.split("measurement")[1])
            # Performing the index renumbering depending on the value of 'driftspresent' (renumbering performed if 'driftspresent' equals "False").
            j = i if driftspresent else i - 1
            # Declaring 'unc_meas' variable as an instantiation of the 'unceratinty_intervals' object.
            unc_meas = uncertainty_intervals(NVFx=NVFx, NVFy=NVFy, NVFz=NVFz, delta_l=delta_l, delta_s=delta_s, delta_p=delta_p)
            # Declaring 'unc_T' variable for storing the uncertainty of the ambient temperature basic measurand. Setting it to 'delta_T'.
            unc_T = unc_meas.delta_T
            # Declaring 'unc_P' variable for storing the uncertainty of the barometric pressure basic measurand. Setting it to 'delta_P'.
            unc_P = unc_meas.delta_P
            # Declaring 'unc_RH' variable for storing the uncertainty of the relative humidity basic measurand. Setting it to 'delta_RH'.
            unc_RH = unc_meas.delta_RH
            # Declaring 'unc_U' variable for storing the uncertainty of the uncorrected velocity measurand. Setting it to 'delta_U'.
            unc_U = unc_meas.delta_U
            # Declaring 'unc_L' variable for storing the uncertainty of the characteristic length. Setting it to 'delta_l'.
            unc_l = unc_meas.delta_l
            # Declaring 'unc_S' variable for storing the uncertainty of the characteristic span. Setting it to 'delta_s'.
            unc_s = unc_meas.delta_s
            # Declaring 'unc_p' variable for storing the uncertainty of the pressure measurements. Setting it to 'delta_p'.
            unc_p = unc_meas.delta_p
            # Declaring 'T_ave' variable for storing average value of measurement temperature. Setting it to the value of its homologous referential magnitude.            
            T_ave = self.__ref_mags__[j].temp - 273
            # Declaring 'P_ave' variable for storing average value of measurement pressure. Setting it to the value of its homologous referential magnitude.
            P_ave = self.__ref_mags__[j].p
            # Declaring 'RH_ave' variable for storing average value of measurement humidity. Setting it to the value of its homologous referential magnitude.
            RH_ave = self.__ref_mags__[j].rh/100
            # Declaring 'Ucorr_ave' variable for storing average value of corrected velocity. Setting it to the value of its homologous referential magnitude.
            Ucorr_ave = self.__ref_mags__[j].u
            # Obtaining slopes and factors of the velocity correction.
            mT, mP, corr = mt.fix_probe_correction(P_ave, T_ave)
            # Computing uncorrected velocity for its introduction in formulas of uncertainty analysis.
            U_ave = Ucorr_ave/corr
            # Declaring 'q_ave' variable for storing average value of upstream dynamic pressure. Setting it to the value of its homologous referential magnitude.
            q_ave = self.__ref_mags__[j].q
            # Declaring 'l' variable for storing average value of characteristic length. Setting it to the value of its homologous referential magnitude.
            l = self.__ref_mags__[j].length
            # Declaring 's' variable for storing average value of characteristic span. Setting it to the value of its homologous referential magnitude.
            s = self.__ref_mags__[j].span
            # Declaring 't' variable for storing average value of characteristic thickness. Setting it to the value of its homologous referential magnitude.
            t = self.__ref_mags__[j].thick
            # Computing characteristic area as the product of characteristic length and span.
            A = l*s
            # Declaring 'rho_ave' variable for storing average value of ambient density. Setting it to the value of its homologous referential magnitude.
            rho_ave = self.__ref_mags__[j].rho
            # Declaring 'mu_ave' variable for storing average value of ambient viscosity. Setting it to the value of its homologous referential magnitude.
            mu_ave = self.__ref_mags__[j].mu
            # Declaring 'angle' variable for storing the value of the current angular position.
            angle = self.__getattribute__(attr).__getattribute__(measurement).angle_pos
            # Conditional inline for storing the value of average x- and y-axis loads and Cl and Cd coefficients in case load measurements have been
            # performed in the current measurement.
            if hasattr(self.__getattribute__(attr).__getattribute__(measurement).channels, "corrected_kistler_fx_signal"):
                # Angle of attack.
                angle = self.__getattribute__(attr).__getattribute__(measurement).angle_pos                
                # x- and y-axes loads.
                fx_ave = np.average(self.__getattribute__(attr).__getattribute__(measurement).channels.__getattribute__("corrected_kistler_fx_signal").data)
                fy_ave = np.average(self.__getattribute__(attr).__getattribute__(measurement).channels.__getattribute__("corrected_kistler_fy_signal").data)
                # y-axis average load (lift): fx*cos(a) - fy*sin(a)
                lift_ave = fx_ave*np.cos(angle*np.pi/180) - fy_ave*np.sin(angle*np.pi/180)
                # x-axis average load (drag): fx*sin(a) + fy*cos(a)
                drag_ave = fx_ave*np.sin(angle*np.pi/180) + fy_ave*np.cos(angle*np.pi/180)
                # Lift coefficient.
                Cl = lift_ave/(q_ave*A)
                # Drag coefficient.
                Cd = drag_ave/(q_ave*A)
            # Otherwise, setting values to NaN.
            else:
                lift_ave = np.NaN
                drag_ave = np.NaN
                Cd = np.NaN
                Cl = np.NaN

            # Conditional for checking whether ambient condition measurements have been made.
            if hasattr(self.__getattribute__(attr).__getattribute__(measurement).channels, "ambient_temperature_signal"):
                ### If they are present, then compute uncertainty deltas of density and viscosity.
                ## Ambient density and viscosity: computing uncertainty intervals.
                # Computing derivatives of density and viscosity with respect to temperature, pressure and relative humidity.
                drho_dT, drho_dP, drho_dRH, dmu_dT, dmu_dP, dmu_dRH = mt.moist_air_density_and_viscosity(P_ave, T_ave, RH_ave, mode='der')
                # Computing uncertainty interval of density.
                unc_rho = np.sqrt((drho_dT*unc_T)**2 + (drho_dP*unc_P)**2 + (drho_dRH*unc_RH)**2)              
                # Computing contributions to density.
                rho_contribs = [(drho_dT*unc_T)**2/unc_rho**2, (drho_dP*unc_P)**2/unc_rho**2, (drho_dRH*unc_RH)**2/unc_rho**2]
                # Setting the 'unc_meas' variable's density delta to the computed uncertainty interval of density.
                unc_meas.delta_rho = unc_rho                  
                # Setting the 'unc_meas' variable's density delta contributions to the computed contributions.
                unc_meas.contrib_T_delta_rho = rho_contribs[0]
                unc_meas.contrib_P_delta_rho = rho_contribs[1]
                unc_meas.contrib_RH_delta_rho = rho_contribs[2]
                # Computing uncertainty interval of viscosity.
                unc_mu = np.sqrt((dmu_dT*unc_T)**2 + (dmu_dP*unc_P)**2 + (dmu_dRH*unc_RH)**2)               
                # Computing contributions to viscosity.
                mu_contribs = [(dmu_dT*unc_T)**2/unc_mu**2, (dmu_dP*unc_P)**2/unc_mu**2, (dmu_dRH*unc_RH)**2/unc_mu**2]
                # Setting the 'unc_meas' variable's viscosity delta to the computed uncertainty interval of viscosity.
                unc_meas.delta_mu = unc_mu
                # Setting the 'unc_meas' variable's viscosity delta contributions to the computed contributions.
                unc_meas.contrib_T_delta_mu = mu_contribs[0]
                unc_meas.contrib_P_delta_mu = mu_contribs[1]
                unc_meas.contrib_RH_delta_mu = mu_contribs[2]                
            else:
                # Otherwise, set ambient variable uncertainty deltas to NaN.
                unc_meas.delta_T = np.NaN
                unc_meas.delta_RH = np.NaN
                unc_meas.delta_P = np.NaN
                unc_meas.delta_rho = np.NaN
                unc_meas.contrib_T_delta_rho = np.NaN
                unc_meas.contrib_P_delta_rho = np.NaN
                unc_meas.contrib_RH_delta_rho = np.NaN
                unc_rho = np.NaN
                unc_meas.delta_mu = np.NaN
                unc_meas.contrib_T_delta_mu = np.NaN
                unc_meas.contrib_P_delta_mu = np.NaN
                unc_meas.contrib_RH_delta_mu = np.NaN
                unc_mu = np.NaN

            # Conditional for checking whether velocity measurements have been made.
            if hasattr(self.__getattribute__(attr).__getattribute__(measurement).channels, "fix_probe_signal"):
                ## If they are present, then perform uncertainty calculations for corrected velocity and dynamic pressure.
                # Computing derivative of corrected velocity with respect to measured velocity.
                dUcorr_dU = (1 + mT*(T_ave - 16))*(1 + mP*(P_ave/100 - 1013.25))
                # Computing derivative of corrected velocity with respect to ambient temperature.
                dUcorr_dT = U_ave*mT*(1 + mP*(P_ave/100 - 1013.25))
                # Computing derivative of corrected velocity with respect to barometric pressure.
                dUcorr_dP = U_ave*mP*(1 + mT*(T_ave - 16))
                # Computing uncertainty interval.
                unc_Ucorr = np.sqrt((dUcorr_dU*unc_U)**2 + (dUcorr_dT*unc_T)**2 + (dUcorr_dP*unc_P/100)**2)
                # Computing contributions.               
                U_corr_contribs = [(dUcorr_dU*unc_U)**2/unc_Ucorr**2, (dUcorr_dT*unc_T)**2/unc_Ucorr**2, (dUcorr_dP*unc_P/100)**2/unc_Ucorr**2]                
                # Setting the 'unc_meas' variable's corrected velocity delta to the computed uncertainty interval.
                unc_meas.delta_U_corr = unc_Ucorr
                # Setting the 'unc_meas' variable's corrected velocity delta contributions to the computed contributions.
                unc_meas.contrib_U_delta_U_corr = U_corr_contribs[0]
                unc_meas.contrib_T_delta_U_corr = U_corr_contribs[1]
                unc_meas.contrib_P_delta_U_corr = U_corr_contribs[2]
                ## Dynamic pressure: computing uncertainty interval.
                # Computing derivative of dynamic pressure with respect to density.
                dq_drho = 0.5*Ucorr_ave**2
                # Computing derivative of dynamic pressure with respect to corrected velocity.
                dq_dUcorr = rho_ave*Ucorr_ave
                # Computing uncertainty interval of dynamic pressure.
                unc_q = np.sqrt((dq_drho*unc_rho)**2 + (dq_dUcorr*unc_Ucorr)**2)
                # Computing contributions.
                q_contribs = [(dq_drho*unc_rho)**2/unc_q**2, (dq_dUcorr*unc_Ucorr)**2/unc_q**2]
                # Setting the 'unc_meas' variable's dynamic pressure delta to the computed uncertainty interval of dynamic pressure.
                unc_meas.delta_q = unc_q
                # Setting the 'unc_meas' variable's dynamic pressure delta contributions to the computed contributions.
                unc_meas.contrib_rho_delta_q = q_contribs[0]
                unc_meas.contrib_U_corr_delta_q = q_contribs[1]
            else:
                # Otherwise, set uncertainty deltas of uncorrected and corrected velocities to NaN.
                unc_meas.delta_U = np.NaN
                unc_meas.contrib_U_delta_U_corr = np.NaN
                unc_meas.contrib_T_delta_U_corr = np.NaN
                unc_meas.contrib_P_delta_U_corr = np.NaN
                unc_meas.delta_U_corr = np.NaN
                unc_meas.contrib_rho_delta_q = np.NaN
                unc_meas.contrib_U_corr_delta_q = np.NaN
                unc_meas.delta_q = np.NaN
                unc_q = np.NaN

            # Conditional for checking whether Reynolds number uncertainty delta is to be calculated.
            if not np.isnan(unc_meas.delta_rho) and not np.isnan(unc_meas.delta_U_corr):
                ## Reynolds number: computing uncertainty interval.
                # Computing derivative of Reynolds number with respect to density.
                dRe_drho = Ucorr_ave*l/mu_ave
                # Computing derivative of Reynolds number with respect to velocity.
                dRe_dUcorr = rho_ave*l/mu_ave
                # Computing derivative of Reynolds number with respect to characteristic length dimension.
                dRe_dl = rho_ave*Ucorr_ave/mu_ave
                # Computing derivative of Reynolds number with respect to viscosity.
                dRe_dmu = -rho_ave*Ucorr_ave*l/mu_ave**2
                # Computing uncertainty interval of Reynolds number.
                unc_Re = np.sqrt((dRe_drho*unc_rho)**2 + (dRe_dUcorr*unc_Ucorr)**2 + (dRe_dl*unc_l)**2 + (dRe_dmu*unc_mu)**2)
                # Computing contributions to Reynolds number.
                Re_contribs = [(dRe_drho*unc_rho)**2/unc_Re**2, (dRe_dUcorr*unc_Ucorr)**2/unc_Re**2, (dRe_dl*unc_l)**2/unc_Re**2, (dRe_dmu*unc_mu)**2/unc_Re**2]
                # Setting the 'unc_meas' variable's Reynolds number delta to the computed uncertainty interval of Reynolds number.
                unc_meas.delta_Re = unc_Re
                # Setting the 'unc_meas' variable's Reynolds number delta contributions to the computed contributions.
                unc_meas.contrib_rho_delta_Re = Re_contribs[0]
                unc_meas.contrib_U_corr_delta_Re = Re_contribs[1]
                unc_meas.contrib_c_delta_Re = Re_contribs[2]
                unc_meas.contrib_mu_delta_Re = Re_contribs[3]                
            else:
                # Otherwise, set uncertainty delta of Reynolds number to NaN.
                unc_meas.delta_Re = np.NaN
                unc_meas.contrib_rho_delta_Re = np.NaN
                unc_meas.contrib_U_corr_delta_Re = np.NaN
                unc_meas.contrib_c_delta_Re = np.NaN
                unc_meas.contrib_mu_delta_Re = np.NaN              

            ## Characteristic area: computing uncertainty interval.
            # Computing derivative of area with respect to length.
            dA_dl = s
            # Computing derivative of area with respect to span.
            dA_ds = l
            # Computing uncertainty interval of characteristic area.
            unc_A = np.sqrt((dA_dl*unc_l)**2 + (dA_ds*unc_s)**2)
            # Setting the 'unc_meas' variable's characteristic area delta to the computed uncertainty interval of characteristic area.
            unc_meas.delta_A = unc_A

            ## Conditional for checking whether lift/drag measurements have been made.
            if not np.isnan(lift_ave):
                ## Uncorrected lift coefficient:computing uncertainty interval.
                # Computing derivative of lift coefficient with respect to Fy force.
                dCl_dFy = np.cos(np.pi*angle/180)/(q_ave*A)
                # Computing derivative of lift coefficient with respect to Fx force.
                dCl_dFx = np.sin(np.pi*angle/180)/(q_ave*A)
                # Computing derivative of lift coefficient with respect to upstream dynamic pressure.
                dCl_dq = -lift_ave/((q_ave**2)*A)
                # Computing derivative of lift coefficient with respect to characteristic area.
                dCl_dA = -lift_ave/(q_ave*A**2)
                # Computing the uncertainty interval of uncorrected lift coefficeint.
                unc_Cl = np.sqrt((dCl_dFy*unc_meas.delta_Fy)**2 + (dCl_dFx*unc_meas.delta_Fx)**2 + (dCl_dq*unc_q)**2 + (dCl_dA*unc_A)**2) if not np.isnan(unc_q) else np.sqrt((dCl_dFy*unc_meas.delta_Fy)**2 + (dCl_dFx*unc_meas.delta_Fx)**2 + (dCl_dA*unc_A)**2)
                # Computing contributions to lift coefficient.
                Cl_contribs = [(dCl_dFy*unc_meas.delta_Fy)**2/unc_Cl**2, (dCl_dFx*unc_meas.delta_Fx)**2/unc_Cl**2, (dCl_dA*unc_A)**2/unc_Cl**2, (dCl_dq*unc_q)**2/unc_Cl**2] if not np.isnan(unc_q) else [(dCl_dFy*unc_meas.delta_Fy)**2/unc_Cl**2, (dCl_dFx*unc_meas.delta_Fx)**2/unc_Cl**2, (dCl_dA*unc_A)**2/unc_Cl**2]
                # Setting the 'unc_meas' variable's uncorrected lift coefficient delta to the computed uncertainty interval of uncorrected lift coefficient.
                unc_meas.delta_Cl = unc_Cl
                # Setting the 'unc_meas' variable's uncorrected lift coefficient delta contributions to the computed contributions.
                unc_meas.contrib_Fy_delta_Cl = Cl_contribs[0]
                unc_meas.contrib_Fx_delta_Cl = Cl_contribs[1]
                unc_meas.contrib_A_delta_Cl = Cl_contribs[2]
                unc_meas.contrib_q_delta_Cl = Cl_contribs[3] if len(Cl_contribs) == 4 else np.NaN
                ## Uncorrected drag coefficient:computing uncertainty interval.
                # Computing derivative of drag coefficient with respect to Fy force.
                dCd_dFy = -np.sin(np.pi*angle/180)/(q_ave*A)
                # Computing derivative of drag coefficient with respect to Fx force.
                dCd_dFx = np.cos(np.pi*angle/180)/(q_ave*A)
                # Computing derivative of drag coefficient with respect to upstream dynamic pressure.
                dCd_dq = -drag_ave/((q_ave**2)*A)
                # Computing derivative of drag coefficient with respect to characteristic area.
                dCd_dA = -drag_ave/(q_ave*A**2)
                # Computing the uncertainty interval of uncorrected drag coefficeint.
                unc_Cd_kis = np.sqrt((dCd_dFy*unc_meas.delta_Fy)**2 + (dCd_dFx*unc_meas.delta_Fx)**2 + (dCd_dq*unc_q)**2 + (dCd_dA*unc_A)**2) if not np.isnan(unc_q) else np.sqrt((dCd_dFy*unc_meas.delta_Fy)**2 + (dCd_dFx*unc_meas.delta_Fx)**2 + (dCd_dA*unc_A)**2)
                # Computing contributions to drag coefficient.
                Cd_contribs = [(dCd_dFy*unc_meas.delta_Fy)**2/unc_Cd_kis**2, (dCd_dFx*unc_meas.delta_Fx)**2/unc_Cd_kis**2, (dCd_dA*unc_A)**2/unc_Cd_kis**2, (dCd_dq*unc_q)**2/unc_Cd_kis**2] if not np.isnan(unc_q) else [(dCd_dFy*unc_meas.delta_Fy)**2/unc_Cd_kis**2, (dCd_dFx*unc_meas.delta_Fx)**2/unc_Cd_kis**2, (dCd_dA*unc_A)**2/unc_Cd_kis**2]
                # Setting the 'unc_meas' variable's uncorrected drag coefficient delta to the computed uncertainty interval of uncorrected lift coefficient.
                unc_meas.delta_Cd_kis = unc_Cd_kis
                # Setting the 'unc_meas' variable's uncorrected drag coefficient delta contributions to the computed contributions.
                unc_meas.contrib_Fy_delta_Cd = Cd_contribs[0]
                unc_meas.contrib_Fx_delta_Cd = Cd_contribs[1]
                unc_meas.contrib_A_delta_Cd = Cd_contribs[2]
                unc_meas.contrib_q_delta_Cd = Cd_contribs[3] if len(Cd_contribs) == 4 else np.NaN         
                ###
                ### Correction parameter computation.
                ###
                ## Geometric parameters.
                # Wind tunnel height value.
                H = 0.902
                # Wind tunnel span value.
                S = 0.75
                # C area constant, equaling the product of wind tunnel height and span.
                C = H * S
                # K1 constant, equaling 0.52 for an airfoil spanning the tunnel's height.
                K1 = 0.52
                # V volume constant, equaling 0.7 times the product of the characteristic dimensions.
                V = 0.7 * t * l * s
                ## Flow-dependent parameters.
                # Solid blockage.
                esb = (K1 * V) / C**(1.5)
                # Wake blockage.
                wb = Cd * (l / (2 * H))
                # Blockage.
                eb = esb + wb
                # Sigma.
                sigma = (np.pi**2 / 48) * (l / H)**2
                ###
                ### Sigma parameter: computing uncertainty interval.
                ###
                # Computing derivative of sigma parameter with respect to characteristic length.
                dsigma_dl = (np.pi**2 / 24) * (l / H**2)
                # Computing uncertainty interval of sigma parameter.
                unc_sigma = np.sqrt((dsigma_dl * unc_l)**2)
                # Setting the 'unc_meas' variable's sigma parameter delta to the computed uncertainty interval of Sigma
                # parameter.
                unc_meas.delta_sigma = unc_sigma
                ###
                ### Solid blockage parameter: computing uncertainty interval.
                ###
                # Computing derivative of solid blockage parameter with respect to characteristic length.
                desb_dl = K1 * 0.7 * t * s / C**1.5
                # Computing derivative of solid blockage parameter with respect to characteristic span.
                desb_ds = K1 * 0.7 * t * l / C**1.5
                # Computing uncertainty interval of solid blockage parameter.
                unc_esb = np.sqrt((desb_dl*unc_l)**2 + (desb_ds*unc_s)**2)
                # Setting the 'unc_meas' variable's solid blockage parameter delta to the computed uncertainty interval of solid
                # blockage parameter.
                unc_meas.delta_esb = unc_esb
                ###
                ### Blockage parameter: computing uncertainty interval.
                ###
                # Computing derivative of blockage parameter with respect to characteristic length.
                deb_dl = desb_dl + Cd/(2*H)
                # Computing derivative of blockage parameter with respect to characteristic span.
                deb_ds = desb_ds
                # Computing derivative of blockage parameter with respect to uncorrected drag coefficient.
                deb_dCd = l / (2 * H)
                # Computing uncertainty interval of blockage parameter.
                unc_eb = np.sqrt((deb_dl*unc_l)**2 + (deb_ds*unc_s)**2 + (deb_dCd*unc_Cd_kis)**2)
                # Setting the 'unc_meas' variable's blockage parameter delta to the computed uncertainty interval of blockage parameter.
                unc_meas.delta_eb = unc_eb 
                ###
                ### Corrected lift coefficient: computing uncertainty interval.
                ###                
                # Computing derivative of corrected lift coefficient with respect to uncorrected lift coefficient.
                dClcorr_dCl = (1 - sigma)/(1 + eb)**2
                # Computing derivative of corrected lift coefficient with respect to sigma parameter.
                dClcorr_dsigma = -Cl/(1 + eb)**2
                # Computing derivative of corrected lift coefficient with respect to blockage parameter.
                dClcorr_deb = -(2*Cl*(1 - sigma))/(1 + eb)**3
                # Computing uncertainty interval of corrected lift coefficient.
                unc_Clcorr = np.sqrt((dClcorr_dCl*unc_Cl)**2 + (dClcorr_dsigma*unc_sigma)**2 + (dClcorr_deb*unc_eb)**2)
                # Computing contributions to corrected lift coefficient.
                Clcorr_contribs = [(dClcorr_dCl*unc_Cl)**2/unc_Clcorr**2, (dClcorr_dsigma*unc_sigma)**2/unc_Clcorr**2, (dClcorr_deb*unc_eb)**2/unc_Clcorr**2]
                # Setting the 'unc_meas' variable's corrected lift coefficient delta to the computed uncertainty interval of corrected lift coefficient.
                unc_meas.delta_Clcorr_kis = unc_Clcorr
                # Setting the 'unc_meas' variable's corrected lift coefficient delta contributions to the computed contributions.
                unc_meas.contrib_Cl_delta_Clcorr_kis = Clcorr_contribs[0]
                unc_meas.contrib_sigma_delta_Clcorr_kis = Clcorr_contribs[1]
                unc_meas.contrib_eb_delta_Clcorr_kis = Clcorr_contribs[2]
                ###
                ### Corrected drag coefficient: computing uncertainty interval.
                ###
                # Computing derivative of corrected drag coefficient with respect to uncorrected drag coefficient.
                dCdcorr_dCd = (1 - esb)/(1 + eb)**2
                # Computing derivative of corrected drag coefficient with respect to solid blockage parameter.
                dCdcorr_desb = -Cd/(1 + eb)**2
                # Computing derivative of corrected drag coefficient with respect to blockage parameter.
                dCdcorr_deb = -(2*Cd*(1 - esb))/(1 + eb)**3
                # Computing uncertainty interval of corrected drag coefficient.
                unc_Cdcorr_kis = np.sqrt((dCdcorr_dCd*unc_Cd_kis)**2 + (dCdcorr_desb*unc_esb)**2 + (dCdcorr_deb*unc_eb)**2)
                # Computing contributions to corrected drag coefficient.
                Cdcorr_contribs = [(dCdcorr_dCd*unc_Cd_kis)**2/unc_Cdcorr_kis**2, (dCdcorr_desb*unc_esb)**2/unc_Cdcorr_kis**2, (dCdcorr_deb*unc_eb)**2/unc_Cdcorr_kis**2]
                # Setting the 'unc_meas' variable's corrected drag coefficient delta to the computed uncertainty interval of corrected drag coefficient.
                unc_meas.delta_Cdcorr_kis = unc_Cdcorr_kis
                # Setting the 'unc_meas' variable's corrected drag coefficient delta contributions to the computed contributions.
                unc_meas.contrib_Cd_delta_Cdcorr_kis = Cdcorr_contribs[0]
                unc_meas.contrib_esb_delta_Cdcorr_kis = Cdcorr_contribs[1]
                unc_meas.contrib_eb_delta_Cdcorr_kis = Cdcorr_contribs[2]
            # Otherwise, setting load and uncorrected Cl and Cd deltas to NaN.
            else:
                unc_meas.delta_Fx = np.NaN
                unc_meas.delta_Fy = np.NaN
                unc_meas.delta_Fz = np.NaN
                unc_meas.delta_Cl = np.NaN
                unc_meas.contrib_Fy_delta_Cl = np.NaN
                unc_meas.contrib_Fx_delta_Cl = np.NaN
                unc_meas.contrib_A_delta_Cl = np.NaN
                unc_meas.contrib_q_delta_Cl = np.NaN
                unc_meas.delta_Cd_kis = np.NaN
                unc_meas.contrib_Fy_delta_Cd = np.NaN
                unc_meas.contrib_Fx_delta_Cd = np.NaN
                unc_meas.contrib_A_delta_Cd = np.NaN
                unc_meas.contrib_q_delta_Cd = np.NaN
                unc_meas.delta_Clcorr_kis = np.NaN
                unc_meas.contrib_Cl_delta_Clcorr_kis = np.NaN
                unc_meas.contrib_sigma_delta_Clcorr_kis = np.NaN
                unc_meas.contrib_eb_delta_Clcorr_kis = np.NaN
                unc_meas.delta_Cdcorr_kis = np.NaN
                unc_meas.contrib_Cd_delta_Cdcorr_kis = np.NaN
                unc_meas.contrib_esb_delta_Cdcorr_kis = np.NaN
                unc_meas.contrib_eb_delta_Cdcorr_kis = np.NaN 

            ## Conditional for checking whether wake-rake measurements have been made.
            if hasattr(self.groups_added, "wake_rake_group"):
                # Computing number of instances for which wake-rake measurements have been performed.
                wake_rake_measurements = len(self.groups_added.means_group.means_channels.__getattribute__([prop for prop in dir(self.groups_added.means_group.means_channels) if "port" in prop][0]))
                # Obtaining number of deficit curves completed during the test.
                deficit_curves = len(self.groups_added.wake_rake_group.angle.y.y_rake_values)
                # Obtaining number of wake-rake measurements per deficit curve as a ratio of 'measurements' and 'deficit_curves' variables.
                measurements_per_deficit_curve = wake_rake_measurements // deficit_curves
                # Defining 'e' index for computing uncertainty interval of drag coefficient for the overall set of measurements comprising a given deficit curve.
                e = (i - 1)//measurements_per_deficit_curve
                # Definint 'e_aux' index for detecting the measurement that starts a new deficit curve.
                e_aux = (i - 2)//measurements_per_deficit_curve if i != 1 else -1
                # Conditional for detecting the initial measurement of a new deficit curve, thus computing the uncertainty interval of the drag coefficient once
                # for each deficit curve, instead of computing it for each measurement comprising that deficit curve.
                if e != e_aux:
                    ###
                    ### Uncorrected drag coefficient: computing uncertainty interval.
                    ###
                    # Getting minimum and maximum indices of the wake of the filtered deficit curve.
                    min_max_wake_indices = self.groups_added.wake_rake_group.angle.y.y_min_max_wake_indices[e]
                    # Getting the length (in points) of the deficit curve.
                    deficit_curve_length = len(self.groups_added.wake_rake_group.angle.y.y_rake_int_values[e])
                    # Getting the filtered deficit curve.
                    deficit_curve = self.groups_added.wake_rake_group.angle.y.y_rake_int_values[e]                    
                    # Getting the standard deviation of the points that lie outside of the wake in the unfiltered deficit curve, so that the filtering
                    # uncertainty can be included in the uncertainty analysis computation.
                    std = np.array([0 for _ in range(0, min_max_wake_indices[0])] + [self.groups_added.wake_rake_group.angle.y.y_discarded_data_stds[e] for _ in range(min_max_wake_indices[0], min_max_wake_indices[1])] + [0 for _ in range(min_max_wake_indices[1], deficit_curve_length)])
                    # Computing the average pressure value of the points in the wake from their respective temporal signals so that the partial
                    # derivatives of the deficit curve with respect to pressure and dynamic pressure can be computed and included in the uncertainty analysis.
                    ps_ave = np.array([np.average(_) for _ in self.groups_added.wake_rake_group.angle.y.y_rake_temp_values[e][min_max_wake_indices[0]:min_max_wake_indices[1]]])
                    # Conditional for checking whether turbulence-related measurements (ps_ave < 0) have been performed; if so, then switch the values to positive so that further
                    # square roots can be taken.
                    if all([_ < 0 for _ in ps_ave]):
                        ps_ave = [np.abs(_) for _ in ps_ave]
                    ## Uncorrected drag coefficient: computing uncertainty interval.
                    # Computing derivative of drag coefficient with respect to pressure (momentum-deficit).
                    dCd_dps = np.array([0 for _ in range(0, min_max_wake_indices[0])] + list((1/q_ave)*((1/(2*np.sqrt(ps_ave/q_ave))) - 1)) + [0 for _ in range(min_max_wake_indices[1], deficit_curve_length)])
                    # Computing derivative of drag coefficient with respect to upstream dynamic pressure.
                    dCd_dqs = np.array([0 for _ in range(0, min_max_wake_indices[0])] + list((ps_ave/q_ave**2)*(1 - (1/(2*np.sqrt(ps_ave/q_ave))))) + [0 for _ in range(min_max_wake_indices[1], deficit_curve_length)])
                    # Computing the uncertainty intervals of the points of the wake. The inline conditional stands for cases that do not own dnamic pressure
                    # uncertainty intervals (i.e. ambient density interval not calculated due to measurements performed before the acquisition of ambient
                    # conditions transmitter).                  
                    unc_Cds = np.sqrt((dCd_dps*unc_p)**2 + (dCd_dqs*unc_q)**2 + std**2) if not np.isnan(unc_q) else np.sqrt((dCd_dps*unc_p)**2 + std**2)
                    # Obtaining wake rake positions for the integration process.
                    rake_pos = np.array([_/(self.__ref_mags__[i].length*1000) for _ in self.groups_added.wake_rake_group.angle.y.y_rake_pos])
                    # Obtaining actual Cd.
                    Cd = self.groups_added.wake_rake_group.angle.angle_cd[e]                    
                    # Obtaining maximum Cd uncertainty value.
                    max_unc_Cd = 2*scipy.integrate.simps(deficit_curve + unc_Cds, rake_pos) - Cd
                    # Obtaining minimum Cd uncertainty value.
                    min_unc_Cd = Cd - 2*scipy.integrate.simps(deficit_curve - unc_Cds, rake_pos)
                    # Setting the Cd uncertainty value to the average of the maximum and minimum values.
                    unc_Cd_wake = np.average([min_unc_Cd, max_unc_Cd])
                    # Computing contributions to Cd.
                    unc_Cds = np.array([_ if _ != 0 else 1 for _ in unc_Cds])
                    Cd_contribs = [(dCd_dps*unc_p)**2/unc_Cds**2, (dCd_dqs*unc_q)**2/unc_Cds**2, std**2/unc_Cds**2] if not np.isnan(unc_q) else [(dCd_dps*unc_p)**2/unc_Cds**2, std**2/unc_Cds**2]
                    # Setting the 'unc_meas' variable's uncorrected drag coefficient delta to the computed uncertainty interval of uncorrected drag coefficient.
                    unc_meas.delta_Cd_wake = unc_Cd_wake
                    # Setting the 'unc_meas' variable's uncorrected drag coefficient delta contributions to the computed contributions.
                    unc_meas.contrib_p_delta_Cd = Cd_contribs[0]
                    unc_meas.contrib_q_wake_delta_Cd = Cd_contribs[1] if not np.isnan(unc_q) else np.NaN
                    unc_meas.contrib_filt_delta_Cd = Cd_contribs[2] if not np.isnan(unc_q) else Cd_contribs[1]
                    ###
                    ### Correction parameter computation.
                    ###
                    ## Geometric parameters.
                    # Wind tunnel height value.
                    H = 0.902
                    # Wind tunnel span value.
                    S = 0.75
                    # C area constant, equaling the product of wind tunnel height and span.
                    C = H * S
                    # K1 constant, equaling 0.52 for an airfoil spanning the tunnel's height.
                    K1 = 0.52
                    # V volume constant, equaling 0.7 times the product of the characteristic dimensions.
                    V = 0.7 * t * l * s
                    ## Flow-dependent parameters.
                    # Solid blockage.
                    esb = (K1 * V) / C**(1.5)
                    # Wake blockage.
                    wb = Cd * (l / (2 * H))
                    # Blockage.
                    eb = esb + wb
                    # Sigma.
                    sigma = (np.pi**2 / 48) * (l / H)**2
                    ###
                    ### Sigma parameter: computing uncertainty interval.
                    ###
                    # Computing derivative of sigma parameter with respect to characteristic length.
                    dsigma_dl = (np.pi**2 / 24) * (l / H**2)
                    # Computing uncertainty interval of sigma parameter.
                    unc_sigma = np.sqrt((dsigma_dl * unc_l)**2)
                    # Setting the 'unc_meas' variable's sigma parameter delta to the computed uncertainty interval of Sigma
                    # parameter.
                    unc_meas.delta_sigma = unc_sigma
                    ###
                    ### Solid blockage parameter: computing uncertainty interval.
                    ###
                    # Computing derivative of solid blockage parameter with respect to characteristic length.
                    desb_dl = K1 * 0.7 * t * s / C**1.5
                    # Computing derivative of solid blockage parameter with respect to characteristic span.
                    desb_ds = K1 * 0.7 * t * l / C**1.5
                    # Computing uncertainty interval of solid blockage parameter.
                    unc_esb = np.sqrt((desb_dl*unc_l)**2 + (desb_ds*unc_s)**2)
                    # Setting the 'unc_meas' variable's solid blockage parameter delta to the computed uncertainty interval of solid
                    # blockage parameter.
                    unc_meas.delta_esb = unc_esb
                    ###
                    ### Blockage parameter: computing uncertainty interval.
                    ###
                    # Computing derivative of blockage parameter with respect to characteristic length.
                    deb_dl = desb_dl + Cd/(2*H)
                    # Computing derivative of blockage parameter with respect to characteristic span.
                    deb_ds = desb_ds
                    # Computing derivative of blockage parameter with respect to uncorrected drag coefficient.
                    deb_dCd = l / (2 * H)
                    # Computing uncertainty interval of blockage parameter.
                    unc_eb = np.sqrt((deb_dl*unc_l)**2 + (deb_ds*unc_s)**2 + (deb_dCd*unc_Cd_wake)**2)
                    # Setting the 'unc_meas' variable's blockage parameter delta to the computed uncertainty interval of blockage parameter.
                    unc_meas.delta_eb = unc_eb
                    ###
                    ### Corrected lift coefficient: computing uncertainty interval.
                    ###
                    # Conditional for checking whether current measurement comprises load measurements.
                    if hasattr(self.groups_added, "kistler_group") and hasattr(self.__getattribute__(attr).__getattribute__(measurement).channels, "corrected_kistler_fy_signal"):
                        # Computing derivative of corrected lift coefficient with respect to uncorrected lift coefficient.
                        dClcorr_dCl = (1 - sigma)/(1 + eb)**2
                        # Computing derivative of corrected lift coefficient with respect to sigma parameter.
                        dClcorr_dsigma = -Cl/(1 + eb)**2
                        # Computing derivative of corrected lift coefficient with respect to blockage parameter.
                        dClcorr_deb = -(2*Cl*(1 - sigma))/(1 + eb)**3
                        # Computing uncertainty interval of corrected lift coefficient.
                        unc_Clcorr = np.sqrt((dClcorr_dCl*unc_Cl)**2 + (dClcorr_dsigma*unc_sigma)**2 + (dClcorr_deb*unc_eb)**2)
                        # Computing contributions to corrected lift coefficient.
                        Clcorr_contribs = [(dClcorr_dCl*unc_Cl)**2/unc_Clcorr**2, (dClcorr_dsigma*unc_sigma)**2/unc_Clcorr**2, (dClcorr_deb*unc_eb)**2/unc_Clcorr**2]
                        # Setting the 'unc_meas' variable's corrected lift coefficient delta to the computed uncertainty interval of corrected lift coefficient.
                        unc_meas.delta_Clcorr_wake = unc_Clcorr
                        # Setting the 'unc_meas' variable's corrected lift coefficient delta contributions to the computed contributions.
                        unc_meas.contrib_Cl_delta_Clcorr_wake = Clcorr_contribs[0]
                        unc_meas.contrib_sigma_delta_Clcorr_wake = Clcorr_contribs[1]
                        unc_meas.contrib_eb_delta_Clcorr_wake = Clcorr_contribs[2]
                    # Otherwise, setting Cl deltas to NaN.
                    else:
                        unc_meas.delta_Cl = np.NaN
                        unc_meas.contrib_F_delta_Cl = np.NaN
                        unc_meas.contrib_A_delta_Cl = np.NaN
                        unc_meas.contrib_q_delta_Cl = np.NaN
                        unc_meas.delta_Clcorr_wake = np.NaN
                        unc_meas.contrib_Cl_delta_Clcorr_wake = np.NaN
                        unc_meas.contrib_sigma_delta_Clcorr_wake = np.NaN
                        unc_meas.contrib_eb_delta_Clcorr_wake = np.NaN          
                    ###
                    ### Corrected drag coefficient: computing uncertainty interval.
                    ###
                    # Computing derivative of corrected drag coefficient with respect to uncorrected drag coefficient.
                    dCdcorr_dCd = (1 - esb)/(1 + eb)**2
                    # Computing derivative of corrected drag coefficient with respect to solid blockage parameter.
                    dCdcorr_desb = -Cd/(1 + eb)**2
                    # Computing derivative of corrected drag coefficient with respect to blockage parameter.
                    dCdcorr_deb = -(2*Cd*(1 - esb))/(1 + eb)**3
                    # Computing uncertainty interval of corrected drag coefficient.
                    unc_Cdcorr_wake = np.sqrt((dCdcorr_dCd*unc_Cd_wake)**2 + (dCdcorr_desb*unc_esb)**2 + (dCdcorr_deb*unc_eb)**2)
                    # Computing contributions to corrected drag coefficient.
                    Cdcorr_contribs = [(dCdcorr_dCd*unc_Cd_wake)**2/unc_Cdcorr_wake**2, (dCdcorr_desb*unc_esb)**2/unc_Cdcorr_wake**2, (dCdcorr_deb*unc_eb)**2/unc_Cdcorr_wake**2]
                    # Setting the 'unc_meas' variable's corrected drag coefficient delta to the computed uncertainty interval of corrected drag coefficient.
                    unc_meas.delta_Cdcorr_wake = unc_Cdcorr_wake
                    # Setting the 'unc_meas' variable's corrected drag coefficient delta contributions to the computed contributions.
                    unc_meas.contrib_Cd_delta_Cdcorr_wake = Cdcorr_contribs[0]
                    unc_meas.contrib_esb_delta_Cdcorr_wake = Cdcorr_contribs[1]
                    unc_meas.contrib_eb_delta_Cdcorr_wake = Cdcorr_contribs[2]
                # Conditional for detecting remaining measurements pertaining the current deficit curve, for setting their uncorrected drag
                # coefficient uncertainty interval to the one corresponding to the present curve.
                if e == e_aux:
                    # Setting the 'unc_meas' variable's uncorrected drag coefficient delta to the computed uncertainty interval of uncorrected
                    # drag coefficient.
                    unc_meas.delta_Cd_wake = unc_Cd_wake
                    # Setting the 'unc_meas' variable's uncorrected drag coefficient delta contributions to the computed contributions.
                    unc_meas.contrib_p_delta_Cd = Cd_contribs[0]
                    unc_meas.contrib_q_wake_delta_Cd = Cd_contribs[1] if not np.isnan(unc_q) else np.NaN
                    unc_meas.contrib_filt_delta_Cd = Cd_contribs[2] if not np.isnan(unc_q) else Cd_contribs[1]                    
                    # Setting the 'unc_meas' variable's sigma parameter delta to the computed uncertainty interval of Sigma
                    # parameter.
                    unc_meas.delta_sigma = unc_sigma
                    # Setting the 'unc_meas' variable's solid blockage parameter delta to the computed uncertainty interval of solid
                    # blockage parameter.
                    unc_meas.delta_esb = unc_esb
                    # Setting the 'unc_meas' variable's blockage parameter delta to the computed uncertainty interval of blockage parameter.
                    unc_meas.delta_eb = unc_eb
                    # Conditional for checking whether current measurement comprises load measurements.
                    if hasattr(self.groups_added, "kistler_group") and hasattr(self.__getattribute__(attr).__getattribute__(measurement).channels, "corrected_kistler_fy_signal"):
                        # Setting the 'unc_meas' variable's corrected lift coefficient delta to the computed uncertainty interval of corrected
                        # lift coefficient.
                        unc_meas.delta_Clcorr_wake = unc_Clcorr
                    else:
                        # Otherwise, setting Clcorr deltas to NaN.
                        unc_meas.delta_Clcorr_wake = np.NaN
                    # Setting the 'unc_meas' variable's corrected drag coefficient delta to the computed uncertainty interval of corrected drag
                    # coefficient.
                    unc_meas.delta_Cdcorr_wake = unc_Cdcorr_wake
                    # Setting the 'unc_meas' variable's corrected drag coefficient delta contributions to the computed contributions.
                    unc_meas.contrib_Cd_delta_Cdcorr_wake = Cdcorr_contribs[0]
                    unc_meas.contrib_esb_delta_Cdcorr_wake = Cdcorr_contribs[1]
                    unc_meas.contrib_eb_delta_Cdcorr_wake = Cdcorr_contribs[2]
            # Otherwise, setting Cd deltas and corrected Cl delta to NaN.
            else:
                unc_meas.delta_Cd_wake = np.NaN
                unc_meas.contrib_p_delta_Cd = np.NaN
                unc_meas.contrib_q_wake_delta_Cd = np.NaN
                unc_meas.contrib_filt_delta_Cd = np.NaN
                unc_meas.delta_Cdcorr_wake = np.NaN
                unc_meas.contrib_Cd_delta_Cdcorr_wake = np.NaN
                unc_meas.contrib_esb_delta_Cdcorr_wake = np.NaN
                unc_meas.contrib_eb_delta_Cdcorr_wake = np.NaN
                unc_meas.delta_Clcorr_wake = np.NaN
                unc_meas.contrib_Cl_delta_Clcorr_wake = np.NaN
                unc_meas.contrib_sigma_delta_Clcorr_wake = np.NaN
                unc_meas.contrib_eb_delta_Clcorr_wake = np.NaN
                
            ## Conditional for checking whether surface pressure measurements have been made.
            if hasattr(self.groups_added, "surface_pressure_group"):
                # Conditional inline for storing the value of average surface pressure measurements in case those measurements have been performed.
                if any([_[:4]=="port" for _ in dir(self.__getattribute__(attr).__getattribute__(measurement).channels)]):
                    # Getting sorted 'port'-dubbed channels (with pressure measurements).
                    portChannels = sorted([portChannel for portChannel in dir(self.__getattribute__(attr).__getattribute__(measurement).channels) if "port" in portChannel], key=lambda x: int(x.split("_")[1]))
                    # Declaring 'ps_ave' variable for storing average values of surface pressure measurements.
                    ps_ave = list()
                    # Declaring 'unc_ps' variable for storing standard deviation values of surface pressure measurements.
                    unc_ps = list()
                    # Loop running over the sorted channels with surface pressure measurements and storing their average values on the 'ps_ave' variable.
                    for port in portChannels:
                        ps_ave.append(np.average(self.__getattribute__(attr).__getattribute__(measurement).channels.__getattribute__(port).data))
                        unc_ps.append(np.std(self.__getattribute__(attr).__getattribute__(measurement).channels.__getattribute__(port).data))
                    ###
                    ### Pressure coefficient: computing uncertainty interval.
                    ###
                    # Computing derivative of pressure coefficient with respect to pressure.
                    dCp_dps = [1/q_ave for _ in portChannels]
                    # Computing derivative of pressure coefficient with respect to dynamic pressure.
                    dCp_dqs = [-p_ave/q_ave**2 for p_ave in ps_ave]
                    # Computing uncertainty interval of pressure coefficient.
                    unc_Cps = [np.sqrt((dCp_dp_dCp_dq_uncs[0]*dCp_dp_dCp_dq_uncs[2])**2 + (dCp_dp_dCp_dq_uncs[1]*unc_q)**2) for dCp_dp_dCp_dq_uncs in zip(dCp_dps, dCp_dqs, unc_ps)]                    
                    # Computing contributions to pressure coefficient.
                    Cp_contribs = [[(dCp_dp_unc_Cp_uncs[0]*dCp_dp_unc_Cp_uncs[2])**2/dCp_dp_unc_Cp_uncs[1]**2 for dCp_dp_unc_Cp_uncs in zip(dCp_dps, unc_Cps, unc_ps)], [(dCp_dq_unc_Cp[0]*unc_q)**2/dCp_dq_unc_Cp[1]**2 for dCp_dq_unc_Cp in zip(dCp_dqs, unc_Cps)]]
                    # Setting the 'unc_meas' variable's pressure coefficient delta to the computed uncertainty interval of pressure coefficient.
                    unc_meas.delta_Cp = unc_Cps
                    # Setting the 'unc_meas' variable's pressure coefficient delta contributions to the computed contributions.
                    unc_meas.contrib_p_delta_Cp = Cp_contribs[0]
                    unc_meas.contrib_q_delta_Cp = Cp_contribs[1]
                # Otherwise, setting values to NaN.
                else:
                    ps_ave = np.NaN
                    unc_meas.delta_Cp = np.NaN
                    unc_meas.contrib_p_delta_Cp = np.NaN
                    unc_meas.contrib_q_delta_Cp = np.NaN
                    
            # Appending computed uncertainty intervals of current measurement to the list of uncertainty intervals.
            unc_list.append(unc_meas)

        # Setting the value of 'unc_list' to the '__unc_ints__' attribute.
        self.__setattr__("__unc_ints__", unc_list)

        # Setting value of __unc_d__ to 'True'.
        self.__unc_d__ = True

        # Non-dimensionalize if necessary.
        if re_dimensionalized:
            self.dimensionalize()

    #######################################################################################################
    ##########################################PUBLIC INSTANCE METHODS######################################
    #######################################################################################################

    # Public method process_standard_data.
    def process_standard_data(self, nd=True):
        """Public method that externally sets the attribute structure pertaining a TDMS file values in case the
        initialization has not done so.

        -**parameters**::
            
        :param nd: boolean flag; if True, the non-dimensionalizing method is run on the results.        
        
        """
    
        # Calling the internal __setStandardData__() method to set the averaged values.
        self.__set_standard_data__(nd=nd)
    
    
    # Public method process_means.
    def process_means(self, notif=True):
        """Public method that externally sets the attribute structure pertaining the averaged values.

        -**parameters**::
            
        :param notif: boolean flag for telling whether notifications are necessary.
        Default is True.        
        
        """              
    
        # Calling the internal __setMeans__() method to set the averaged values.
        self.__set_means__(notif=notif)

    # Public method process_wake_rake.
    def process_wake_rake(self, wake_rake_port_ordering=tuple(np.arange(1, 19)), wake_rake_cross_axis=TDMSE.deviceAxis.y, notif=True):
        """Public method that externally sets the attribute structure pertaining wake rake values.
        
        -**parameters**::
    
        :param wake_rake_cross_axis: an instance of the custom TDMSE.deviceAxis class specifying the traverse direction, relative
        to wind tunnel axes, of the rake.
        :param notif: boolean flag for telling whether notifications are necessary.
        Default is True.        
        
        """     
    
        # Calling the internal __set_wake_rake__ method for performing the task of attribute structuring.
        self.__set_wake_rake__(wake_rake_port_ordering=wake_rake_port_ordering, wake_rake_cross_axis=wake_rake_cross_axis, notif=notif)

    # Public method process_kistler.
    def process_kistler(self, device_axis=TDMSE.deviceAxis.angle, notif=True):
        """Public method that externally sets the attribute structure pertaining kistler values.
        
        -**parameters**::
            
        :param device_axis: an instance of the custom TDMSE.deviceAxis class specifying the driving axis of the experiment.
        :param notif: boolean flag for telling whether notifications are necessary.
        Default is True.        
        
        """          
        
        # Calling the internal __set_kistler__ method for performing the task of attribute structuring.
        self.__set_kistler__(device_axis=device_axis, notif=notif)
        
    # Public method process_cobra.
    def process_cobra(self, device_axis=TDMSE.deviceAxis.y, notif=True):
        """Public method that externally sets the attribute structure pertaining Cobra values.
        
        -**parameters**::
            
        :param device_axis: an instance of the custom TDMSE.deviceAxis class specifying the driving axis of the experiment.
        :param notif: boolean flag for telling whether notifications are necessary.
        Default is True.        
        
        """  
        
        # Calling the internal __set_cobra__ method for performing the task of attribute structuring.
        self.__set_cobra__(device_axis = device_axis, notif=notif)
        
    # Public method process_kistler_wake_rake.
    def process_kistler_wake_rake(self, kistler_device_axis=TDMSE.deviceAxis.angle, wake_rake_port_ordering=tuple(np.arange(1, 19)), wake_rake_device_axis=TDMSE.deviceAxis.y, notif=True):
        """Public method that externally sets the attributes pertaining to Kistler and wake-rake values.
        
        -**parameters**::
            
        :param kistler_device_axis: an instance of the custom TDMSE.deviceAxis class specifying the driving axis of the experiment.
        Default is TDMSE.deviceAxis.angle, meaning that the angle of attack has been employed as axis.
        :param wake_rake_port_ordering: tells the ordering of the scanner ports by which the wake-rake pneumatic lines are arranged.
        Default is range(1, 19), meaning that the first port is 1 and the last one 18.
        :param wake_rake_device_axis: an instance of the custom TDMSE.deviceAxis class specifying the traverse direction, relative
        to wind tunnel axes, of the rake.
        Default is TDMSE.deviceAxis.y, meaning that the y-dimensions (tunnel width) has been employed as axis.
        :param notif: boolean flag for telling whether notifications are necessary.
        Default is True.
        
        """
        
        # Calling the internal __set_kistler__ method for performing the task of Kistler attribute structuring.
        self.__set_kistler__(device_axis=kistler_device_axis, notif=notif)
        # Calling the internal __set_wake_rake__ method for performing the task of wake-rake attribute structuring.
        self.__set_wake_rake__(wake_rake_port_ordering = wake_rake_port_ordering, device_axis=wake_rake_device_axis)
        
    # Public methods process_surface_pressure.
    def process_surface_pressure(self, surface_device_axis=TDMSE.deviceAxis.angle, port_ordering=tuple(np.arange(1, 31)), port_pos=TDMSE.pressure_taps.NACA0021_elec, notif=True):
        """Public method that externally sets the attributes pertaining to surface pressure measurements.
        
        -**parameters**::
            
        :param surface_device_axis: an instance of the custom TDMSE.deviceAxis class specifying the driving axis of the experiment.
        Default is TDMSE.deviceAxis.angle, meaning that the angle of attack has been employed as axis.
        :param port_ordering: tells the ordering of the scanner ports by which surface pressure measurements have been performed.
        Default is range(1, 31), meaning that the first port is 1 and the last one 30.
        :param port_pos: tells the positions of the pressure taps along the body that houses them.
        Default is TDMSE.pressure_taps.NACA0021_elec, meaning that the positions coincide with the straight row of pressure taps practiced
        in the modular NACA0021 airfoil manufactures by wire-cutting.
        :param notif: boolean flag for telling whether notifications are necessary.
        Default is True.
        
        """
        
        # Calling the internal __set_surface_pressure__ method for performing the task of surface pressure attribute structuring.
        self.__set_surface_pressure__(device_axis=surface_device_axis, port_ordering=port_ordering, port_pos=port_pos, notif=notif)

    # Public mehtod dimensionalize.
    def dimensionalize(self, mode='non', notif=True):
        """Performs non/re-dimensionalization on the TDMS file data.

        -**parameters**::

        :param mode: string literal 'non/re' indicating whether a non-dimensionalization or a re-dimensionalization is to be performed.
        Default is 'non'.
        :param notif: boolean flag for telling whether notifications are necessary.
        Default is True.

        """

        # Calling the internal ____ method for performing the task of non/re-dimensionalizing.
        self.__dimensionalize__(self.__file__, mode=mode, notif=notif)
        
    # Public mehtod corrector.
    def corrector(self, mode='kistler'):
        """Performs wall-interference correction of lift/drag curves and fluid variables.
        
        The corrections are performed according to the formulae of 2011Selig.

        -**parameters**::

        :param mode: a string literal, either 'kistler' or 'wakeRake', indicating which drag coefficient
        data is to be employed for correcting the curves (the corrections are Cd-value-dependent).

        """

        # Calling the internal __corrector__ method for performing the task of correcting lift/drag curves.
        self.__corrector__(mode=mode)        

    # Public method uncertainty_analysis.
    def uncertainty_analysis(self, NVFx=5, NVFy=5, NVFz=5, delta_l=1.4e-6, delta_s=0.7e-6, delta_p=7.5e-2):
        """Performs an uncertainty analysis on the TDMS file data setting the uncertainty intervals on the dataset.
        
        -**parameters**

        :param NVFx: N/V scale of the piezoelectric's x-axis. Default is 5.
        :param NVFy: N/V scale of the piezoelectric's y-axis. Default is 5.
        :param NVFz: N/V scale of the piezoelectric's z-axis. Default is 5.
        :param delta_l: uncertainty interval corresponding to the characteristic length. Default is 1.4e-3.
        :param delta_s: uncertainty interval corresponding to the characteristic span. Default is 0.7e-3.
        :param delta_p: uncertainty interval corresponding to pressure scanner measurements. Default is 7.5e-2.

        """

        # Calling the internal __set_uncertainty_intervals__ method for performing the task of setting the uncertainty intervals.
        self.__set_uncertainty_intervals__(NVFx=NVFx, NVFy=NVFy, NVFz=NVFz, delta_l=delta_l, delta_s=delta_s, delta_p=delta_p)

    # Public method isND.
    def isND(self):
        """Tells whether the data of the TDMS file is in a dimensional/non-dimensional format."""

        # Return statement.
        return self.__nd_d__

    # Public method isUnc.
    def isUnc(self):
        """Tells whether the data of the TDMS file has already been uncertainty-analysed."""

        # Return statement.
        return self.__unc_d__

    # Public method refMags.
    def refMags(self, measurements=[0]):
        """Provides the referential magnitudes employed on a given TDMS file.

         -**parameters**::

        :param measurements: a list of indices indicating the measurements whose reference magnitudes are to be shown.
        Default is 0, meaning that the first measurement's reference magnitudes are shown solely.        

        """

        # Declaring 'attr' variable and setting it to 'groups_original' string literal. It intends to store the name of the original group object of the TDMS file.
        attr = "groups_original"
        # Conditional for checking whether the file owns a '__groups_original__' group: if true, then the 'attr' variable is changed accordingly.
        if hasattr(self, "__groups_original__"):
            attr = "__groups_original__"

        # Declaring the 'meas_list' variable and assigning to it the ordered list (according to the measurement index) of the measurements.
        meas_list = sorted([meas for meas in dir(self.__getattribute__(attr)) if "measurement" in meas], key=lambda i: int(i.split("measurement")[1]))
        # Conditional for checking whether drift measurements have been performed. If true, then 'initial_drift' and 'final_drift' string literals
        # are inserted into the 'meas_list' variable.
        if "initial_drift" in dir(self.__getattribute__(attr)):
            meas_list.insert(0, "initial_drift")
            meas_list.insert(len([meas for meas in dir(self.__getattribute__(attr)) if "__" not in meas]), "final_drift")

        dict_params_values = {}
        
        # For loop running over the set of passed measurements and printing its referential magnitudes.
        for measurement in measurements:
            dict_param_value = {}
            # print("Ref magnitudes of " + meas_list[measurement] + ":")
            whole_list = [("length", "m"), ("span", "m"), ("thick", "m"), ("temp", "ºC"), ("p", "Pa"), ("rh", "%"), ("rho", "kg/m^3"), ("mu", "kg/ms"), ("u", "m/s"), ("q", "Pa")]
            param_list = [_[0] for _ in whole_list]
            # For loop for obtaining the referential magnitudes sorted according to the order provided in the 'param_list' variable.
            for _ in enumerate(sorted([_ for _ in dir(self.__ref_mags__[measurement]) if ("__" not in _) and ("r_const" not in _)], key=lambda i: param_list.index(i))):
                # print(_[1], " (", whole_list[_[0]][1], ") =", "{:.2E}".format(self.__ref_mags__[measurement].__getattribute__(_[1])))
                dict_param_value[_[1]] = self.__ref_mags__[measurement].__getattribute__(_[1])
            # if measurements.index(measurement) != len(measurements) - 1:
                # print("\n")
            dict_params_values[meas_list[measurement]] = dict_param_value
            
        # return statement.
        return dict_params_values

    # Public method uncIntervals.
    def uncIntervals(self, measurand=TDMSE.measurands.velcorr):
        """Provides the uncertainty intervals computed for a given TDMS file's dataset.

        -**parameters**::
        
        :param measurands: string indicating whether 'basic' or 'derived' magnitudes are to be shown. Default is 'derived'.

        """

        # Declaring 'attr' variable and setting it to 'groups_original' string literal. It intends to store the name of the original group object of the TDMS file.
        attr = "groups_original"
        # Conditional for checking whether the file owns a '__groups_original__' group: if true, then the 'attr' variable is changed accordingly.
        if hasattr(self, "__groups_original__"):
            attr = "__groups_original__"

        # Declaring the 'meas_list' variable and assigning to it the ordered list (according to the measurement index) of the measurements.
        meas_list = sorted([meas for meas in dir(self.__getattribute__(attr)) if "measurement" in meas], key=lambda i: int(i.split("measurement")[1]))
        
        # Conditional for checking whether drift measurements have been performed. If true, then 'initial_drift' and 'final_drift' string literals
        # are inserted into the 'meas_list' variable.
        # if "initial_drift" in dir(self.__getattribute__(attr)):
        #     meas_list.insert(0, "initial_drift")
        #     meas_list.insert(len([meas for meas in dir(self.__getattribute__(attr)) if "__" not in meas]), "final_drift")

        # Getting measurand attribute to retrieve from TDMSE.
        measurand_attribute = measurand.value
        # Setting 'unc_ints' to the uncertainty interval list of the present file.
        unc_ints = self.__unc_ints__
        
        if len(measurand_attribute) == 1:
            # Casting into dictionary a list comprehension statement setting a tuple with the measurement name and the requested uncertainty value.
            unc_dict = dict([(meas_list[e], unc_ints[e].__getattribute__(measurand_attribute[0])) for e, _ in enumerate(unc_ints) if not np.isnan(unc_ints[e].__getattribute__(measurand_attribute[0]))])
        else:
            # Assigning empty list to 'deltas_dict' variable, which intends to store the deltas and contributions of the requested variable for each measurement.
            deltas_dict = list()
            # Assigning empty list to 'meas_with_deltas' variable, which intends to store the measurements that own non-NaN deltas for the requested variable.
            meas_with_deltas = list()
            # Assigning empty list to 'unc_dict' variable, which intends to store the dictionary of uncertainties (deltas and contributions) for the measurements
            # owning non-NaN deltas and contributions for the requested variable.
            unc_dict = list()
            # 'drifts' variable set to value 1 if drift measurements are performed; 0 otherwise. For keeping measurement index consistency.
            drifts = 1 if "initial_drift" in dir(self.__getattribute__(attr)) else 0
            # 'for' loop running over the measurements for retrieveing the deltas and contributions.
            for e, meas in enumerate(meas_list):
                # Conditional for checking whether the current measurement's requested delta is present (type==list or not NaN), in which case it is retrieved together
                # with the correspondent contributions.
                if (type(unc_ints[e].__getattribute__(measurand_attribute[0])) == list) or not np.isnan(unc_ints[e].__getattribute__(measurand_attribute[0])):
                    # Appending deltas and contributions in a dictionary-like variable to 'deltas_dict' variable.
                    deltas_dict.append(dict([("delta", unc_ints[e].__getattribute__(measurand_attribute[0]))] + [(measurand_attribute[f + 1].split("delta")[0][:-1], unc_ints[e].__getattribute__(measurand_attribute[f + 1])) for f, _ in enumerate(measurand_attribute[1:])]))
                    # Appending index of current measurement to 'meas_with_deltas' variable.
                    meas_with_deltas.append(e + drifts)
            # 'for' loop running over the measurements with non-NaN deltas of the requested variable and appending to 'unc_dict'.
            for e, _ in enumerate(meas_with_deltas):
                unc_dict.append((meas_list[e], deltas_dict[e]))
            # Conditional for checking whether it is necessary to cast the 'unc_dict' list into a dictionary (i.e. whether 'deltas' or uncertainties are to be
            # shown, which is checked by computing the length of the 'unc_dict' variable).
            if len(unc_dict):
                # If the length is not null, then the dictionary is created by adding 'meas_with_deltas' as keys and 'deltas_dict' as values.
                unc_dict = dict([(meas_with_deltas[e], deltas_dict[e]) for e, _ in enumerate(unc_dict)])
        # Return statement.
        return unc_dict

######################################################################################################################
#####################################################PRIVATE CLASSES##################################################
######################################################################################################################

# Internal class __TdmsFileGroups__, which contains the group objects of the tdmsFileRoot object.
class __TdmsFileGroups__:
    """Internal class __tdmsFileGroups__ representing the entity of groups of a TDMS file."""

    # __init__ (constructor) of the internal class __tdmsFileGroups__.
    def __init__(self, file_obj, group_objs=None, generate_isolated_groups=False):
        """Initializes an instance of the __tdmsFileGroups__ object.

        The __tdmsFileGroups__ object is the entity object that contains the groups of a TDMS file.

        - **parameters**, **return**, **return types**::

        :param file_obj: an instantiation of the nptdms.TdmsFile object.
        :param group_objs: a list containing strings of group names to be instantiated as nptdms.GroupObject objects

        :param generate_isolated_groups: boolean flag signaling whether an isolated group object is to be instantiated.
        :return: __tdmsFileGroups__ object structured in accordance to the groups contained on the TDMS file.
        :rtype: __tdmsFileGroups__.
        """

        # Conditional for checking whether an isolated groups is to be instantiated. The code hits this block if the
        # boolean flag is false.
        if group_objs is None:
            group_objs = list()
        if not generate_isolated_groups:
            # Declaring a group_index variable for running over the group objects passed as parameters.
            group_index = 0
            # For loop running over the group objects. For each group object, a __tdmsFileGroup__ object is
            # instantiated and set as an attribute on the parent __tdmsFileGroups__ object.
            for groupObj in group_objs:                
                self.__setattr__(groupObj.name.lower().replace(" ", "_"), __TdmsFileGroup__(groupObj.name, group_index, file_obj, False))
                group_index += 1
        else:
            pass

    # Internal __addIsolatedGroup__ method.
    def __add_isolated_group__(self, isolated_group_name):
        """Instantiates an isolated __TdmsFileGroup__ object.

        The method is a side call to the nptdms library built-in method nptdms.GroupObject(), which is the proper
        instantiator of a group object. This is done so to allow declaring isolated group instantiations from the
        parent group __TdmsFileGroups__. Declaring isolated groups turns necessary when running the root initializer
        with certain reading modes.

        - **parameters**

        :param isolated_group_name: name given to the isolated group to be instantiated.

        """

        # Call to the static internal method __TdmsFileGroup__.__generateIsolatedGroup__() which is, itself, a side
        # call to the proper group instantiator nptdms.GroupObject().
        self.__setattr__(isolated_group_name, __TdmsFileGroup__.__generate_isolated_group__(isolated_group_name))


# Internal class __TdmsFileGroup__, which contains individual groups that populate a __tdmsFileGroups__ object.
class __TdmsFileGroup__:
    """Internal class __TdmsFileGroup__ representing a group entity of a TDMS file."""

    # __init__ (constructor) of the internal class __tdmsFileGroup__.
    def __init__(self, group_obj, group_index, file_obj, generate_isolated_group=False):
        """Initializes an instance of the __TdmsFileGroup__ object.

        The __TdmsFileGroup__ object is the entity object representing a group of a TDMS file.

        - **parameters**, **return**, **return types**::

        :param group_obj: a string literal containing the declared name of the group object to be instantiated.
        :param group_index: index for identifying the specific group within the groups of the fileObj variable and thus
        retrieving the properties of the group for its proper instantiation.
        :param file_obj: an instantiation of the nptdms.TdmsFile object representing a TDMS file.
        :param generate_isolated_group: boolean flag signaling whether an isolated group is to be instantiated.
        :return: __TdmsFileGroup__ object structured in accordance to the properties and channels contained within
        that group.
        :rtype: __TdmsFileGroup__.

        """

        # Conditional for checking whether an isolated group is to be instantiated. The code hits this block if the
        # boolean flag is false.
        if not generate_isolated_group:
            # Conditional for allowing the instantiation of nptdms.GroupObject objects outside the scope of a specific
            # nptmds.TdmsFile object.
            if file_obj is None:
                pass
            else:                
                # Declaring the attributes of "group" and "group_name".
                self.group_name = group_obj                
                # Declaring the list of properties "prop_list".
                prop_list = list((file_obj.groups()[group_index]).properties.items())                
                # A "for" loop running over the properties list and setting the correspondent attributes.
                for prop in prop_list:
                    self.__setattr__(prop[0].lower().replace(" ", "_"), prop[1])
                # Declaring the attribute "channels" as an instantiation of the __TdmsGroupChannels__ object, which
                # represents the collection of channels contained within a group.
                self.channels = __TdmsGroupChannels__(self, file_obj[self.group_name].channels(), generate_isolated_channels=False)                
        else:
            pass

    # Static internal __generateIsolatedGroup__ method.
    @staticmethod
    def __generate_isolated_group__(isolated_group_name):
        """Static internal method __generateIsolatedGroup__ for generating isolated groups.

        The groups are instantiations of the nptdms.GroupObject objects.

        - **parameters**, **returns**, **return types**

        :param isolated_group_name: string literal providing the declaration name of the instantiated object.
        :return: __tdmsFileGroup__ object reflecting a nptdms.GroupObject object.
        :rtype: __TdmsFileGroup__

        """

        # Return statement instantiating a __tdmsFileGroup__ object.
        return __TdmsFileGroup__(nptdms.GroupObject(isolated_group_name), None, None, False)

        # Internal __addIsolatedChannels__ method.

    def __add_isolated_channels__(self, isolated_channels_name):
        """Programmatically declares an entity intended to store the channels coming from a TDMS group.

        The information found at the group level is extended by adding an entity intended to store the channels coming
        from the TDMS group being read. The declared name for this variable is a settable parameter of the method.

        - **parameters**, **return**, **return types**::

        :param isolated_channels_name: name of the programmatically declared variable.
        :return: __tdmsFileChannels__ object.
        :rtype: __tdmsFileChannels__

        """

        # Declaring an instantiation of the __tdmsFileChannels__ object and setting it into the "isolatedChannelsName"
        # variable.
        self.__setattr__(isolated_channels_name, __TdmsGroupChannels__(None, None, generate_isolated_channels=True))


# Internal class __TdmsGroupChannels__, which contains the channel objects of a __tdmsFileGroup__ object.
class __TdmsGroupChannels__:
    """Internal class __TdmsGroupChannels__ representing the entity of channels of a TDMS group."""

    # __init__ (constructor) of the internal class __TdmsGroupChannels__.
    def __init__(self, group_obj, channel_objs, generate_isolated_channels=False):
        """Initializes an instance of the __TdmsGroupChannels__ object.

        The __TdmsGroupChannels__ object is the entity containing the channels of a TDMS group.

        - **parameters**, **return**, **return types**::

        :param group_obj: an instantiation of the nptdms.GroupObject object representing a TDMS group for which the
        channels are to be determined.
        :param channel_objs: a list of string literals containing the names of the channels to be declared as
        instantiations of the __TdmsGroupChannel__ object.
        :param generate_isolated_channels: boolean flag signaling whether an isolated channel is to be instantiated.
        :return: __TdmsGroupChannels__ object structured in accordance to the channels contained on a TDMS group.
        :rtype: __TdmsGroupChannels__.

        """
        
        # Conditional for checking whether an isolated channel is to be instantiated. This block of code is hit solely
        # when the boolean flag is false.
        if not generate_isolated_channels:
            # Declaring the channel_obj looping variable for running over the channel names and declaring them as
            # instantiations of the __TdmsGroupChannel__ object.            
            for channel_obj in channel_objs:
                self.__setattr__(channel_obj.name.lower().replace(" ", "_"), __TdmsGroupChannel__(channel_obj.name.lower().replace(" ", "_"), nptdms.ChannelObject(channel_obj, group_obj, channel_obj.data)))
        else:
            pass

    # Internal __add_isolated_channel__ method.
    def __add_isolated_channel__(self, isolated_channel_name):
        """Instantiates an isolated __TdmsGroupChannel__ object.

        The method is a call to the initializer of the __TdmsGroupChannel__ object.

        - **parameters**

        :param isolated_channel_name: name given to the isolated channel to be instantiated.

        """

        # Call to the static internal method __tdmsGroupchannel__.__generateIsolatedchannel__(), which declares an
        # instantiation of the __TdmsGroupChannel__ object.
        self.__setattr__(isolated_channel_name, __TdmsGroupChannel__.__generate_isolated_channel__(isolated_channel_name))


# Internal class __tdmsFileChannel__, which contains individual channels that populate a __tdmsFileChannels__ object.
class __TdmsGroupChannel__:
    """Internal class __TdmsGroupChannel__ representing a channel entity of a TDMS group."""

    # __init__ (constructor) of the internal class __TdmsGroupChannel__.
    def __init__(self, channel_name, channel_obj, generate_isolated_channel=False):
        """Initializes an instance of the __TdmsGroupChannel__ object.

        The __TdmsGroupChannel__ object is the entity representing a channel of a TDMS group.

        - **parameters**, **return**, **return types**::

        :param channel_name: declared name of the channel to be instantiated.
        :param channel_obj: an instantiation of the nptdms.group_channel object containing a TDMS channel information.
        :param generate_isolated_channel: boolean flag signaling whether an isolated channel is to be instantiated.
        :return: __TdmsGroupChannel__ object structured in accordance to the properties contained within itself.
        :rtype: __TdmsGroupChannel__.

        """

        # Conditional for checking whether an isolated channel is to be instantiated.
        if not generate_isolated_channel:
            # Declaring channel_name, data and dataMean variables in case the boolean flag owns a false value.
            self.channelName = channel_name
            self.data = channel_obj.data           
        else:
            # Declaring channel_name, data (empty array) and dataMean (NaN) in case the boolean flag owns a true value.
            self.channelName = channel_name
            self.data = np.array(())

    # Static internal __generate_isolated_channel__ method.
    @staticmethod
    def __generate_isolated_channel__(isolated_channel_name):
        """Static internal method __generate_isolated_channel__ for generating isolated groups.

        The channels are instantiations of this same object (__tdmsGroupChannel__). This approach is sensible insofar
        it turns necessary to auto-declare certain channels.

        - **parameters**, **returns**, **return types**

        :param isolated_channel_name: string literal providing the declaration name of the instantiated object.
        :return: __tdmsGroupChannel__ object reflecting an instantiation of this same object.
        :rtype: __TdmsGroupChannel__

        """

        # Return statement instantiating a __TdmsGroupChannel__ object.
        return __TdmsGroupChannel__(isolated_channel_name, None, True)


# Internal class __RunsPerParam__, which contains the basic individual structure for constructing the wake rake processor.
class __RunsPerParam__:
    """Internal class __RunsPerParam__, which contains the basic individual structure for constructing the
    attribute-based structure that classifies the data of a file according to its positional information."""

    # __init__ (constructor) of the internal class __RunsPerParam__.
    def __init__(self, name, value=None):
        """Initializes an instance of the __RunsPerParam__ object."""

        # Setting an attribute with the given name to the given value.
        if value is None:
            value = list()
        self.__setattr__(name, value)

######################################################################################################################
######################################################PUBLIC CLASSES##################################################
######################################################################################################################

# Public class XFoilRoot, which contains the basic structure for constructing a data entity coming from an XFoil file.
class XFoilRoot:
    """Public class XFoilRoot representing the core entity of an XFoil file."""

    #####
    ##### Declaring variables.
    ##### 
    # angle info on XFoil file.
    angles = None
    # lift info on XFoil file.
    lift = None
    # drag info on XFoil file.
    drag = None
    # eff info on XFoil file.
    eff = None
    # pitch info on XFoil file.
    pitch = None

    #####
    ##### Defining methods.
    ##### 
    def __init__(self, angles, lift, drag, eff, pitch):
        """Initializes an instance of the XFoilRoot object.

        The XFoilRoot is the object that gathers the information concerning a given dataset
        obtained from XFoil computations.

        - **parameters**, **return**, **return types**::

        :param angles: the dataset of angles-of-attack.
        :param lift: the dataset of lifts.
        :param drag: the dataset of drags.
        :param eff: the dataset of efficiencies.
        :param pitch: the dataset of pitches.
        :return: an instantiated object of the XFoilRoot class.
        :rtype: XFoilRoot.
        """
        
        #####
        ##### Setting attributes.
        ##### 
        self.angles = angles
        self.lift = lift
        self.drag = drag
        self.eff = eff
        self.pitch = pitch

# Public class TdmsFileData, which provides the necessary metadata to read a TDMS file.
class TdmsFileData:
    """Public class TdmsFileData representing the information necessary to open a TDMS file."""

    # Path of the TDMS file to be opened.
    file_path = None
    # Alias to be given to the TdmsFileRoot object instance to be created.
    file_alias = None
    # Reading mode to be employed when opening the file.
    file_read_mode = None
    # Title to be given in the charts to the plots coming from this file.
    file_title = None
    # Instance of the 'RefMagnitudes' object to be used for non-dimensionalizing purposes.
    file_ref_mag = None
    
    #####
    ##### Defining methods.
    ##### 
    def __init__(self, path, alias, title=None, read_mode=TDMSE.tdmsFilereadMode.standard, ref_mag=""):
        """Initializes an instance of the TdmsFileData object.

        The TdmsFileData object is the basic object that addresses a TDMS file to be read.

        - **parameters**, **return**, **return types**::

        :param path: path of the file to be read.
        :param file_alias: string by which the data pertaining to a specific file is recognized.
        :param read_mode: custom enum object specifying the reading mode:
            -standard: reads and damps the TDMS structure, as a whole, to runtime variables.
            -means: reads and damps the TDMS structure to runtime variables, adding to the structure itself
            the average values of the read data.
            -means_only: reads and damps the TDMS structure to runtime variables, considering solely the
            average values of the read data.
            -projected: to be used when force measurements are included; it projects the forces according
            to the measurement angle to provide the loads on wind tunnel axes.
            -projected_means_only: to be used when force measurements are included; it projects the force
            averages according to the measurement angle to provide the average loads on wind tunnel axes.
            -wake_rake: to be used when wake rake measurements are included.        
        Deault is "standard".
        :param ref_mag: an instantiation of the dataclass RefMagnitudes for non-dimensionalizing purposes.
        Default is "".
        :return: TdmsFileData object representing a reference to a TDMS file to be read.
        :rtype: TdmsFileData
        """
        
        #####
        ##### Setting attributes.
        ##### 
        # Path of the file to be read.
        self.file_path = path
        # Alias of the file to be read.
        self.file_alias = alias
        # Title of the file to be read.
        self.file_title = title
        # Reaading mode.
        self.file_read_mode = read_mode
        # Reference magnitudes.
        self.file_ref_mag = ref_mag

# Public class XFoilFileData, which provides the necessary metadata to read a XFoil file.
class XFoilFileData:
    """Public class TdmsFileData representing the information necessary to open a XFoil file."""

    #####
    ##### Declaring magnitudes.
    ##### 
    # Path of the XFoil file to be opened.
    file_path = None
    # Alias to be given to the XFoilRoot object instance to be created.
    file_alias = None
    # Title to be given in the charts to the plots coming from this file.
    file_title = None

    #####
    ##### Defining methods.
    #####   
    def __init__(self, path, alias, title):
        """Initializer of class XFoilFileData.

        - **parameters**, **returns**, **return types**

        :param path: path of XFoil file.
        :param alias: alias by which to name the file.
        :param title: title by which to name the file.
        :return: an instantiated object of the XFoilFileData class.
        :rtype: XFoilFileData.
        
        """
        
        #####
        ##### Setting attributes.
        #####   
        self.file_path = path
        self.file_alias = alias
        self.file_title = title


# Public class RefMagnitudes, which is a C#-like struct class for non-dimensionalizing purposes.
@dtc.dataclass
class RefMagnitudes:
    """Public class for setting referential or characteristic magnitudes for non/re-dimensionalizing purposes."""

    #####
    ##### Declaring internal magnitudes.
    ##### 
    # Internal member characteristic length (chordwise)
    __length__: float = 1
    # Internal member characteristic length (spanwise)
    __span__: float = 1  
    # Internal member characteristic length (thickness)
    __thick__: float = 1  
    # Internal member characteristic velocity
    __u__: float = 1  
    # Internal member characteristic pressure
    __p__: float = 1  
    # Internal member characteristic temperature
    __temp__: float = 1
    # Internal member characteristic relative humidity
    __rh__: float = 1
    # Internal member gas constant
    __r_const__: float = 1  
    # Internal member density
    __rho__: float = dtc.field(init=False)  
    # Internal member viscosity
    __mu__: float = dtc.field(init=False)  
    # Internal member dynamic pressure
    __q__: float = dtc.field(init=False)  

    #####
    ##### Defining methods.
    #####   
    # Internal __post_init__ method for computing derived magnitudes such as density or dynamic pressure.
    def __post_init__(self):
        '''Post initializer function for determining the derived reference magnitudes from the basic ones.'''                
        # Density and viscosity computation by humid air law. Call to external function moist_air_density_and_viscosity().
        self.__rho__, self.__mu__ = mt.moist_air_density_and_viscosity(self.p, self.temp - 273.15, self.rh/100, mode='noder')
        # Dynamic pressure computation.
        self.__q__ = 0.5 * self.rho * self.u * self.u  
   
    # Declaration of get property length for characteristic length (chordwise).
    @property
    def length(self) -> float:
        return self.__length__

    # Declaration of set property length for characteristic length (chordwise).
    @length.setter
    def length(self, value: float) -> None:
        self.__length__ = value

    # Declaration of get property span for characteristic length (spanwise).
    @property
    def span(self) -> float:
        return self.__span__

    # Declaration of set property span for characteristic length (spanwise).
    @span.setter
    def span(self, value: float) -> None:
        self.__span__ = value

    # Declaration of get property thick for characteristic length (thickness).
    @property
    def thick(self) -> float:
        return self.__thick__

    # Declaration of set property thick for characteristic length (thickness).
    @thick.setter
    def thick(self, value: float) -> None:
        self.__thick__ = value

    # Declaration of get property u for characteristic velocity.
    @property
    def u(self) -> float:
        return self.__u__

    # Declaration of set property u for characteristic velocity.
    @u.setter
    def u(self, value: float) -> None:
        self.__u__ = value
        self.__post_init__()

    # Declaration of get property p for characteristic pressure.
    @property
    def p(self) -> float:
        return self.__p__

    # Declaration of set property p for characteristic pressure.
    @p.setter
    def p(self, value: float) -> None:
        self.__p__ = value
        self.__post_init__()

    # Declaration of get property temp for characteristic temperature.
    @property
    def temp(self) -> float:
        return self.__temp__

    # Declaration of set property temp for characteristic temperature.
    @temp.setter
    def temp(self, value: float) -> None:
        self.__temp__ = value
        self.__post_init__()

    # Declaration of get property rh for characteristic relative humidity.
    @property
    def rh(self) -> float:
        return self.__rh__

    # Declaration of set property rh for characteristic relative humidity.
    @rh.setter
    def rh(self, value: float) -> None:
        self.__rh__ = value
        self.__post_init__()

    # Declaration of get property r_const for gas constant.
    @property
    def r_const(self) -> float:
        return self.__r_const__

    # Declaration of set property for gas constant.
    @r_const.setter
    def r_const(self, value: float) -> None:
        self.__r_const__ = value
        self.__post_init__()

    # Declaration of get property rho for derived density.
    @property
    def rho(self) -> float:
        return self.__rho__

    # Declaration of get property mu for derived viscosity.
    @property
    def mu(self) -> float:
        return self.__mu__

    # Declaration of get property q for dervied dynamic pressure.
    @property
    def q(self) -> float:
        return self.__q__
    
    # Initialization method.
    def __init__(self, length=1., span=1., thick=1., vel=0., press=101325., temp=273., rh=60, r_const=287.):
        """Initializer of data class RefMagnitudes.

        By design philosophy, an initializer should not be necessary when employing the dataclass decorator. However,
        the drawback of such a decorator is to lack of a proper signature display on intellisense when instantiating
        the object. The purpose of the custom initializer is to provide users with such an aid when using the object.

        - **parameters**, **returns**, **return types**

        :param length: characteristic length (chordwise, m).
        :param span: characteristic length (spanwise, m).
        :param thick: characteristic length (thickness, m).
        :param vel: characteristic velocity (m/s).
        :param press: characteristic pressure (Pa).
        :param temp: characteristic temperature (ºK).
        :param rh: relative humidity (%).
        :param r_const: characteristic gas constant (J/(kg·ºK)).
        :return: an instantiation of the dataclass refMagnitude object.
        :rtype: refMagnitude.
        
        """
        
        #####
        ##### Declaring attributes.
        #####         
        # Setting length.
        self.__length__ = length
        # Setting span.
        self.__span__ = span
        # Setting thickness.
        self.__thick__ = thick
        # Setting reference velocity.
        self.__u__ = vel
        # Setting reference pressure.
        self.__p__ = press
        # Setting reference temperature.
        self.__temp__ = temp
        # Setting reference relative humidity.
        self.__rh__ = rh
        # Setting gas constant.
        self.__r_const__ = r_const                
        # __post_init__() method call for obtaining derived magnitudes.
        self.__post_init__()
    
# Public class uncertinaty_intervals, which is a C#-like struct class for obtaining uncertainty_intervals.
@dtc.dataclass
class uncertainty_intervals:
    """Public class for setting uncertainty intervals of measurements on a TDMS file data."""

    #####
    ##### Declaring internal variables.
    #####         
    # Ambient temperature delta.
    __delta_T__: float = 2.38e-2
    
    # Barometric pressure delta.
    __delta_P__: float = 4.76e-3*600
    
    # Relative humidity delta.
    __delta_RH__: float = 2.38e-2
    
    # Velocity delta.
    __delta_U__: float = 3.83e-2*(np.sqrt(2000/mt.moist_air_density_and_viscosity(1013*100, 16, 0)[0]))/100
    
    # Fx delta.
    __delta_Fx__: float = 0.44
    
    # Fy delta.
    __delta_Fy__: float = 0.44
    
    # Fz delta.
    __delta_Fz__: float = 0.44
    
    # length delta.
    __delta_l__: float = 1.4e-6
    
    # span delta.
    __delta_s__: float = 0.7e-6
    
    # pressure delta.
    __delta_p__: float = 0
    
    # Corrected velocity delta.
    __delta_U_corr__: float = 0
    # U contribution to velocity delta.
    __contrib_U_delta_U_corr__: float = 0
    # T contribution to velocity delta.
    __contrib_T_delta_U_corr__: float = 0
    # P contribution to velocity delta.
    __contrib_P_delta_U_corr__: float = 0
    
    # rho delta.
    __delta_rho__: float = 0
    # T contribution to rho delta.
    __contrib_T_delta_rho__: float = 0
    # P contribution to rho delta.
    __contrib_P_delta_rho__: float = 0
    # RH contribution to rho delta.
    __contrib_RH_delta_rho__: float = 0
    
    # mu delta.
    __delta_mu__: float = 0
    # T contribution to mu delta.
    __contrib_T_delta_mu__: float = 0
    # P contribution to mu delta.
    __contrib_P_delta_mu__: float = 0
    # RH contribution to mu delta.
    __contrib_RH_delta_mu__: float = 0    
    
    # q delta.
    __delta_q__: float = 0
    # rho contribution to q delta.
    __contrib_rho_delta_q__: float = 0
    # U_corr contribution to q delta.
    __contrib_U_corr_delta_q__: float = 0
    
    # Re delta.
    __delta_Re__: float = 0
    # rho contribution to Re delta.
    __contrib_rho_delta_Re__: float = 0
    # U_corr contribution to Re delta.
    __contrib_U_corr_delta_Re__: float = 0
    # c contribution to Re delta.
    __contrib_c_delta_Re__: float = 0
    # mu contribution to Re delta.
    __contrib_mu_delta_Re__: float = 0
    
    # area delta.
    __delta_A__: float = 0
    
    # Cl delta.
    __delta_Cl__: float = 0
    # Fy contribution to Cl delta.
    __contrib_Fy_delta_Cl__: float = 0
    # Fx contribution to Cl delta.
    __contrib_Fx_delta_Cl__: float = 0    
    # A contribution to Cl delta.
    __contrib_A_delta_Cl__: float = 0
    # q contribution to Cl delta.
    __contrib_q_delta_Cl__: float = 0
    
    # Cd_kis delta.
    __delta_Cd_kis__: float = 0
    # Fx contribution to Cd_kis delta.
    __contrib_Fx_delta_Cd__: float = 0
    # Fy contribution to Cd_kis delta.
    __contrib_Fy_delta_Cd__: float = 0    
    # A contribution to Cd_kis delta.
    __contrib_A_delta_Cd__: float = 0
    # q contribution to Cd_kis delta.
    __contrib_q_delta_Cd__: float = 0    
    
    # Cd_wake delta.
    __delta_Cd_wake__: float = 0
    # p contribution to Cd delta.
    __contrib_p_delta_Cd__: float = 0
    # q contribution to Cd delta.
    __contrib_q_wake_delta_Cd__: float = 0
    # filt contribution to Cd delta.
    __contrib_filt_delta_Cd__: float = 0
    
    # sigma delta.
    __delta_sigma__: float = 0
    
    # blockage delta.
    __delta_eb__: float = 0
    
    # solid blockage delta.
    __delta_esb__: float = 0
    
    # corrected Cl delta from Kistler.
    __delta_Clcorr_kis__: float = 0
    # Cl contribution to Clcorr delta from Kistler.
    __contrib_Cl_delta_Clcorr_kis__: float = 0
    # sigma contribution to Clcorr delta from Kistler.
    __contrib_sigma_delta_Clcorr_kis__: float = 0
    # eb contribution to Clcorr delta from Kistler.
    __contrib_eb_delta_Clcorr_kis__: float = 0
    
    # corrected Cl delta from wake.
    __delta_Clcorr_wake__: float = 0
    # Cl contribution to Clcorr delta from wake.
    __contrib_Cl_delta_Clcorr_wake__: float = 0
    # sigma contribution to Clcorr delta from wake.
    __contrib_sigma_delta_Clcorr_wake__: float = 0
    # eb contribution to Clcorr delta from wake.
    __contrib_eb_delta_Clcorr_wake__: float = 0    
    
    # corrected Cd delta from Kistler.
    __delta_Cdcorr_kis__: float = 0
    # Cd contribution to Cdcorr delta from Kistler.
    __contrib_Cd_delta_Cdcorr_kis__: float = 0
    # esb contribution to Cdcorr delta from Kistler.
    __contrib_esb_delta_Cdcorr_kis__: float = 0
    # eb contribution to Cdcorr delta from Kistler.
    __contrib_eb_delta_Cdcorr_kis__: float = 0
    
    # corrected Cd delta from wake.
    __delta_Cdcorr_wake__: float = 0
    # Cd contribution to Cdcorr delta from wake.
    __contrib_Cd_delta_Cdcorr_wake__: float = 0
    # esb contribution to Cdcorr delta from wake.
    __contrib_esb_delta_Cdcorr_wake__: float = 0
    # eb contribution to Cdcorr delta from wake.
    __contrib_eb_delta_Cdcorr_wake__: float = 0
    
    # Cp delta.
    __delta_Cp__: float = 0
    # p contribution to Cp delta.
    __contrib_p_delta_Cp__: float = 0
    # q contribution to Cp delta.
    __contrib_q_delta_Cp__: float = 0

    #####
    ##### Defining methods.
    #####         
    # Declaration of get property delta_T for ambient temperature delta.
    @property
    def delta_T(self) -> float:
        return self.__delta_T__

    # Declaration of set property delta_T for ambient temperature delta.
    @delta_T.setter
    def delta_T(self, value: float) -> None:
        self.__delta_T__ = value

    # Declaration of get property delta_P for barometric pressure delta.
    @property
    def delta_P(self) -> float:
        return self.__delta_P__

    # Declaration of set property delta_P for barometric pressure delta.
    @delta_P.setter
    def delta_P(self, value: float) -> None:
        self.__delta_P__ = value

    # Declaration of get property delta_RH for relative humidity delta.
    @property
    def delta_RH(self) -> float:
        return self.__delta_RH__

    # Declaration of set property delta_RH for relative humidity delta.
    @delta_RH.setter
    def delta_RH(self, value: float) -> None:
        self.__delta_RH__ = value

    # Declaration of get property delta_U for velocity delta.
    @property
    def delta_U(self) -> float:
        return self.__delta_U__

    # Declaration of set property delta_U for velocity delta.
    @delta_U.setter
    def delta_U(self, value: float) -> None:
        self.__delta_U__ = value

    # Declaration of get property delta_Fx for Fx delta.
    @property
    def delta_Fx(self) -> float:
        return self.__delta_Fx__

    # Declaration of set property delta_Fx for Fx delta.
    @delta_Fx.setter
    def delta_Fx(self, value: float) -> None:
        self.__delta_Fx__ = value

    # Declaration of get property delta_Fy for Fy delta.
    @property
    def delta_Fy(self) -> float:
        return self.__delta_Fy__

    # Declaration of set property delta_Fy for Fy delta.
    @delta_Fy.setter
    def delta_Fy(self, value: float) -> None:
        self.__delta_Fy__ = value

    # Declaration of get property delta_Fz for Fz delta.
    @property
    def delta_Fz(self) -> float:
        return self.__delta_Fz__

    # Declaration of set property delta_Fz for Fz delta.
    @delta_Fz.setter
    def delta_Fz(self, value: float) -> None:
        self.__delta_Fz__ = value

    # Declaration of get property delta_l for characteristic length delta.
    @property
    def delta_l(self) -> float:
        return self.__delta_l__

    # Declaration of set property delta_l for characteristic length delta.
    @delta_l.setter
    def delta_l(self, value: float) -> None:
        self.__delta_l__ = value

    # Declaration of get property delta_s for characteristic span delta.
    @property
    def delta_s(self) -> float:
        return self.__delta_s__

    # Declaration of set property delta_s for characteristic span delta.
    @delta_s.setter
    def delta_s(self, value: float) -> None:
        self.__delta_s__ = value

    # Declaration of get property delta_p for pressure delta.
    @property
    def delta_p(self) -> float:
        return self.__delta_p__

    # Declaration of set property delta_p for pressure delta.
    @delta_p.setter
    def delta_p(self, value: float) -> None:
        self.__delta_p__ = value

    # Declaration of get property delta_U_corr for corrected velocity delta.
    @property
    def delta_U_corr(self) -> float:
        return self.__delta_U_corr__

    # Declaration of set property delta_U_corr for corrected velocity delta.
    @delta_U_corr.setter
    def delta_U_corr(self, value: float) -> None:
        self.__delta_U_corr__ = value
        
    # Declaration of get property contrib_U_delta_U_corr for contribution of velocity to corrected velocity delta.
    @property
    def contrib_U_delta_U_corr(self) -> float:
        return self.__contrib_U_delta_U_corr__
    
    # Declaration of set property contrib_U_delta_U_corr for contribution of velocity to corrected velocity delta.
    @contrib_U_delta_U_corr.setter
    def contrib_U_delta_U_corr(self, value: float) -> None:
        self.__contrib_U_delta_U_corr__ = value
        
    # Declaration of get property contrib_T_delta_U_corr for contribution of temperature to corrected velocity delta.
    @property
    def contrib_T_delta_U_corr(self) -> float:
        return self.__contrib_T_delta_U_corr__
    
    # Declaration of set property contrib_T_delta_U_corr for contribution of temperature to corrected velocity delta.
    @contrib_T_delta_U_corr.setter
    def contrib_T_delta_U_corr(self, value: float) -> None:
        self.__contrib_T_delta_U_corr__ = value       
        
    # Declaration of get property contrib_P_delta_U_corr for contribution of pressure to corrected velocity delta.
    @property
    def contrib_P_delta_U_corr(self) -> float:
        return self.__contrib_P_delta_U_corr__
    
    # Declaration of set property contrib_P_delta_U_corr for contribution of pressure to corrected velocity delta.
    @contrib_P_delta_U_corr.setter
    def contrib_P_delta_U_corr(self, value: float) -> None:
        self.__contrib_P_delta_U_corr__ = value            

    # Declaration of get property delta_rho for ambient density delta.
    @property
    def delta_rho(self) -> float:
        return self.__delta_rho__
    
    # Declaration of set property delta_rho for ambient density delta.
    @delta_rho.setter
    def delta_rho(self, value: float) -> None:
        self.__delta_rho__ = value
    
    # Declaration of get property contrib_T_delta_rho for contribution of temperature to density delta.
    @property
    def contrib_T_delta_rho(self) -> float:
        return self.__contrib_T_delta_rho__
    
    # Declaration of set property contrib_T_delta_rho for contribution of temperature to density delta.
    @contrib_T_delta_rho.setter
    def contrib_T_delta_rho(self, value: float) -> None:
        self.__contrib_T_delta_rho__ = value

    # Declaration of get property contrib_P_delta_rho for contribution of pressure to density delta.
    @property
    def contrib_P_delta_rho(self) -> float:
        return self.__contrib_P_delta_rho__
    
    # Declaration of set property contrib_P_delta_rho for contribution of pressure to density delta.
    @contrib_P_delta_rho.setter
    def contrib_P_delta_rho(self, value: float) -> None:
        self.__contrib_P_delta_rho__ = value

    # Declaration of get property contrib_RH_delta_rho for contribution of RH to density delta.
    @property
    def contrib_RH_delta_rho(self) -> float:
        return self.__contrib_RH_delta_rho__
    
    # Declaration of set property contrib_RH_delta_rho for contribution of RH to density delta.
    @contrib_RH_delta_rho.setter
    def contrib_RH_delta_rho(self, value: float) -> None:
        self.__contrib_RH_delta_rho__ = value                 

    # Declaration of get property delta_mu for ambient viscosity delta.
    @property
    def delta_mu(self) -> float:
        return self.__delta_mu__

    # Declaration of set property delta_mu for ambient viscosity delta.
    @delta_mu.setter
    def delta_mu(self, value: float) -> None:
        self.__delta_mu__ = value 
       
    # Declaration of get property contrib_T_delta_mu for contribution of temperature to viscosity delta.
    @property
    def contrib_T_delta_mu(self) -> float:
        return self.__contrib_T_delta_mu__
    
    # Declaration of set property contrib_T_delta_mu for contribution of temperature to viscosity delta.
    @contrib_T_delta_mu.setter
    def contrib_T_delta_mu(self, value: float) -> None:
        self.__contrib_T_delta_mu__ = value

    # Declaration of get property contrib_P_delta_mu for contribution of pressure to viscosity delta.
    @property
    def contrib_P_delta_mu(self) -> float:
        return self.__contrib_P_delta_mu__
    
    # Declaration of set property contrib_P_delta_mu for contribution of pressure to viscosity delta.
    @contrib_P_delta_mu.setter
    def contrib_P_delta_mu(self, value: float) -> None:
        self.__contrib_P_delta_mu__ = value

    # Declaration of get property contrib_RH_delta_mu for contribution of RH to viscosity delta.
    @property
    def contrib_RH_delta_mu(self) -> float:
        return self.__contrib_RH_delta_mu__
    
    # Declaration of set property contrib_RH_delta_mu for contribution of RH to viscosity delta.
    @contrib_RH_delta_mu.setter
    def contrib_RH_delta_mu(self, value: float) -> None:
        self.__contrib_RH_delta_mu__ = value

    # Declaration of get property delta_q for dynamic pressure delta
    @property
    def delta_q(self) -> float:
        return self.__delta_q__
    
    # Declaration of set property delta_q for dynamic pressure delta.
    @delta_q.setter
    def delta_q(self, value: float) -> None:
        self.__delta_q__ = value    
    
    # Declaration of get property contrib_rho_delta_q for contribution of rho to q delta.
    @property
    def contrib_rho_delta_q(self) -> float:
        return self.__contrib_rho_delta_q__
    
    # Declaration of set property contrib_rho_delta_q for contribution of rho to q delta.
    @contrib_rho_delta_q.setter
    def contrib_rho_delta_q(self, value: float) -> None:
        self.__contrib_rho_delta_q__ = value

    # Declaration of get property contrib_U_corr_delta_q for contribution of U_corr to q delta.
    @property
    def contrib_U_corr_delta_q(self) -> float:
        return self.__contrib_U_corr_delta_q__
    
    # Declaration of set property contrib_U_corr_delta_q for contribution of U_corr to q delta.
    @contrib_U_corr_delta_q.setter
    def contrib_U_corr_delta_q(self, value: float) -> None:
        self.__contrib_U_corr_delta_q__ = value  

    # Declaration of get property delta_s for Reynolds number delta.
    @property
    def delta_Re(self) -> float:
        return self.__delta_Re__

    # Declaration of set property delta_s for Reynolds number delta.
    @delta_Re.setter
    def delta_Re(self, value: float) -> None:
        self.__delta_Re__ = value

    # Declaration of get property contrib_rho_delta_Re for contribution of rho to Re delta.
    @property
    def contrib_rho_delta_Re(self) -> float:
        return self.__contrib_rho_delta_Re__
    
    # Declaration of set property contrib_rho_delta_Re for contribution of rho to Re delta.
    @contrib_rho_delta_Re.setter
    def contrib_rho_delta_Re(self, value: float) -> None:
        self.__contrib_rho_delta_Re__ = value

    # Declaration of get property contrib_U_corr_delta_Re for contribution of U_corr to Re delta.
    @property
    def contrib_U_corr_delta_Re(self) -> float:
        return self.__contrib_U_corr_delta_Re__
    
    # Declaration of set property contrib_U_corr_delta_Re for contribution of U_corr to Re delta.
    @contrib_U_corr_delta_Re.setter
    def contrib_U_corr_delta_Re(self, value: float) -> None:
        self.__contrib_U_corr_delta_Re__ = value

    # Declaration of get property contrib_c_delta_Re for contribution of c to Re delta.
    @property
    def contrib_c_delta_Re(self) -> float:
        return self.__contrib_c_delta_Re__
    
    # Declaration of set property contrib_c_delta_Re for contribution of c to Re delta.
    @contrib_c_delta_Re.setter
    def contrib_c_delta_Re(self, value: float) -> None:
        self.__contrib_c_delta_Re__ = value

    # Declaration of get property contrib_mu_delta_Re for contribution of mu to Re delta.
    @property
    def contrib_mu_delta_Re(self) -> float:
        return self.__contrib_mu_delta_Re__
    
    # Declaration of set property contrib_mu_delta_Re for contribution of mu to Re delta.
    @contrib_mu_delta_Re.setter
    def contrib_mu_delta_Re(self, value: float) -> None:
        self.__contrib_mu_delta_Re__ = value               

    # Declaration of get property delta_s for characteristic area delta.
    @property
    def delta_A(self) -> float:
        return self.__delta_A__

    # Declaration of set property delta_s for characteristic area delta.
    @delta_A.setter
    def delta_A(self, value: float) -> None:
        self.__delta_A__ = value

    # Declaration of get property delta_Cl for uncorrected lift coefficient delta.
    @property
    def delta_Cl(self) -> float:
        return self.__delta_Cl__

    # Declaration of set property delta_Cl for uncorrected lift coefficient delta.
    @delta_Cl.setter
    def delta_Cl(self, value: float) -> None:
        self.__delta_Cl__ = value
        
    # Declaration of get property contrib_Fy_delta_Cl for contribution of Fy to Cl delta.
    @property
    def contrib_Fy_delta_Cl(self) -> float:
        return self.__contrib_Fy_delta_Cl__
    
    # Declaration of set property contrib_Fy_delta_Cl for contribution of Fy to Cl delta.
    @contrib_Fy_delta_Cl.setter
    def contrib_Fy_delta_Cl(self, value: float) -> None:
        self.__contrib_Fy_delta_Cl__ = value
        
    # Declaration of get property contrib_Fx_delta_Cl for contribution of Fx to Cl delta.
    @property
    def contrib_Fx_delta_Cl(self) -> float:
        return self.__contrib_Fx_delta_Cl__
    
    # Declaration of set property contrib_Fx_delta_Cl for contribution of Fx to Cl delta.
    @contrib_Fx_delta_Cl.setter
    def contrib_Fx_delta_Cl(self, value: float) -> None:
        self.__contrib_Fx_delta_Cl__ = value        
        
    # Declaration of get property contrib_A_delta_Cl for contribution of A to Cl delta.
    @property
    def contrib_A_delta_Cl(self) -> float:
        return self.__contrib_A_delta_Cl__
    
    # Declaration of set property contrib_A_delta_Cl for contribution of A to Cl delta.
    @contrib_A_delta_Cl.setter
    def contrib_A_delta_Cl(self, value: float) -> None:
        self.__contrib_A_delta_Cl__ = value
        
    # Declaration of get property contrib_q_delta_Cl for contribution of q to Cl delta.
    @property
    def contrib_q_delta_Cl(self) -> float:
        return self.__contrib_q_delta_Cl__
    
    # Declaration of set property contrib_q_delta_Cl for contribution of q to Cl delta.
    @contrib_q_delta_Cl.setter
    def contrib_q_delta_Cl(self, value: float) -> None:
        self.__contrib_q_delta_Cl__ = value
        
    # Declaration of get property delta_Cd for uncorrected lift coefficient delta.
    @property
    def delta_Cd_kis(self) -> float:
        return self.__delta_Cd_kis__

    # Declaration of set property delta_Cd_kis for uncorrected lift coefficient delta.
    @delta_Cd_kis.setter
    def delta_Cd_kis(self, value: float) -> None:
        self.__delta_Cd_kis__ = value
        
    # Declaration of get property contrib_Fx_delta_Cd for contribution of Fx to Cl delta.
    @property
    def contrib_Fx_delta_Cd(self) -> float:
        return self.__contrib_Fx_delta_Cd__
    
    # Declaration of set property contrib_Fx_delta_Cd for contribution of Fx to Cl delta.
    @contrib_Fx_delta_Cd.setter
    def contrib_Fx_delta_Cd(self, value: float) -> None:
        self.__contrib_Fx_delta_Cd__ = value
        
    # Declaration of get property contrib_Fy_delta_Cd for contribution of Fy to Cl delta.
    @property
    def contrib_Fy_delta_Cd(self) -> float:
        return self.__contrib_Fy_delta_Cd__
    
    # Declaration of set property contrib_Fy_delta_Cd for contribution of Fy to Cl delta.
    @contrib_Fy_delta_Cd.setter
    def contrib_Fy_delta_Cd(self, value: float) -> None:
        self.__contrib_Fy_delta_Cd__ = value        
        
    # Declaration of get property contrib_A_delta_Cd for contribution of A to Cl delta.
    @property
    def contrib_A_delta_Cd(self) -> float:
        return self.__contrib_A_delta_Cd__
    
    # Declaration of set property contrib_A_delta_Cd for contribution of A to Cl delta.
    @contrib_A_delta_Cd.setter
    def contrib_A_delta_Cd(self, value: float) -> None:
        self.__contrib_A_delta_Cd__ = value
        
    # Declaration of get property contrib_q_delta_Cd for contribution of q to Cl delta.
    @property
    def contrib_q_delta_Cd(self) -> float:
        return self.__contrib_q_delta_Cd__
    
    # Declaration of set property contrib_q_delta_Cd for contribution of q to Cl delta.
    @contrib_q_delta_Cd.setter
    def contrib_q_delta_Cd(self, value: float) -> None:
        self.__contrib_q_delta_Cd__ = value        

    # Declaration of get property delta_Cd_wake for uncorrected drag coefficient delta.
    @property
    def delta_Cd_wake(self) -> float:
        return self.__delta_Cd_wake__

    # Declaration of set property delta_Cd_wake for uncorrected drag coefficient delta.
    @delta_Cd_wake.setter
    def delta_Cd_wake(self, value: float) -> None:
        self.__delta_Cd_wake__ = value
        
    # Declaration of get property contrib_p_delta_Cd for contribution of p to Cd delta.
    @property
    def contrib_p_delta_Cd(self) -> float:
        return self.__contrib_p_delta_Cd__
    
    # Declaration of set property contrib_p_delta_Cd for contribution of p to Cd delta.
    @contrib_p_delta_Cd.setter
    def contrib_p_delta_Cd(self, value: float) -> None:
        self.__contrib_p_delta_Cd__ = value
        
    # Declaration of get property contrib_q_wake_delta_Cd for contribution of q to Cd delta.
    @property
    def contrib_q_wake_delta_Cd(self) -> float:
        return self.__contrib_q_wake_delta_Cd__
    
    # Declaration of set property contrib_q_wake_delta_Cd for contribution of q to Cd delta.
    @contrib_q_wake_delta_Cd.setter
    def contrib_q_wake_delta_Cd(self, value: float) -> None:
        self.__contrib_q_wake_delta_Cd__ = value
        
    # Declaration of get property contrib_filt_delta_Cd for contribution of filt to Cd delta.
    @property
    def contrib_filt_delta_Cd(self) -> float:
        return self.__contrib_filt_delta_Cd__
    
    # Declaration of set property contrib_filt_delta_Cd for contribution of filt to Cd delta.
    @contrib_filt_delta_Cd.setter
    def contrib_filt_delta_Cd(self, value: float) -> None:
        self.__contrib_filt_delta_Cd__ = value

    # Declaration of get property delta_sigma for solidity delta.
    @property
    def delta_sigma(self) -> float:
        return self.__delta_sigma__

    # Declaration of set property delta_sigma for solidity delta.
    @delta_sigma.setter
    def delta_sigma(self, value: float) -> None:
        self.__delta_sigma__ = value

    # Declaration of get property delta_eb for blockage delta.
    @property
    def delta_eb(self) -> float:
        return self.__delta_eb__

    # Declaration of set property delta_eb for blockage delta.
    @delta_eb.setter
    def delta_eb(self, value: float) -> None:
        self.__delta_eb__ = value

    # Declaration of get property delta_esb for solid blockage delta.
    @property
    def delta_esb(self) -> float:
        return self.__delta_esb__

    # Declaration of set property delta_esb for solid blockage delta.
    @delta_esb.setter
    def delta_esb(self, value: float) -> None:
        self.__delta_esb__ = value

    # Declaration of get property delta_Clcorr_kis for corrected lift coefficient delta.
    @property
    def delta_Clcorr_kis(self) -> float:
        return self.__delta_Clcorr_kis__

    # Declaration of set property delta_Clcorr_kis for corrected lift coefficient delta.
    @delta_Clcorr_kis.setter
    def delta_Clcorr_kis(self, value: float) -> None:
        self.__delta_Clcorr_kis__ = value
        
    # Declaration of get property delta_Clcorr_wake for corrected lift coefficient delta.
    @property
    def delta_Clcorr_wake(self) -> float:
        return self.__delta_Clcorr_wake__

    # Declaration of set property delta_Clcorr_wake for corrected lift coefficient delta.
    @delta_Clcorr_wake.setter
    def delta_Clcorr_wake(self, value: float) -> None:
        self.__delta_Clcorr_wake__ = value
        
    # Declaration of get property contrib_Cl_delta_Clcorr_kis for contribution of Cl to Clcorr delta.
    @property
    def contrib_Cl_delta_Clcorr_kis(self) -> float:
        return self.__contrib_Cl_delta_Clcorr_kis__
    
    # Declaration of set property contrib_Cl_delta_Clcorr_kis for contribution of Cl to Clcorr delta.
    @contrib_Cl_delta_Clcorr_kis.setter
    def contrib_Cl_delta_Clcorr_kis(self, value: float) -> None:
        self.__contrib_Cl_delta_Clcorr_kis__ = value
        
    # Declaration of get property contrib_Cl_delta_Clcorr_wake for contribution of Cl to Clcorr delta.
    @property
    def contrib_Cl_delta_Clcorr_wake(self) -> float:
        return self.__contrib_Cl_delta_Clcorr_wake__
    
    # Declaration of set property contrib_Cl_delta_Clcorr_wake for contribution of Cl to Clcorr delta.
    @contrib_Cl_delta_Clcorr_wake.setter
    def contrib_Cl_delta_Clcorr_wake(self, value: float) -> None:
        self.__contrib_Cl_delta_Clcorr_wake__ = value        
        
    # Declaration of get property contrib_sigma_delta_Clcorr_kis for contribution of sigma to Clcorr delta.
    @property
    def contrib_sigma_delta_Clcorr_kis(self) -> float:
        return self.__contrib_sigma_delta_Clcorr_kis__
    
    # Declaration of set property contrib_sigma_delta_Clcorr_kis for contribution of sigma to Clcorr delta.
    @contrib_sigma_delta_Clcorr_kis.setter
    def contrib_sigma_delta_Clcorr_kis(self, value: float) -> None:
        self.__contrib_sigma_delta_Clcorr_kis__ = value
        
    # Declaration of get property contrib_sigma_delta_Clcorr_wake for contribution of sigma to Clcorr delta.
    @property
    def contrib_sigma_delta_Clcorr_wake(self) -> float:
        return self.__contrib_sigma_delta_Clcorr_wake__
    
    # Declaration of set property contrib_sigma_delta_Clcorr_wake for contribution of sigma to Clcorr delta.
    @contrib_sigma_delta_Clcorr_wake.setter
    def contrib_sigma_delta_Clcorr_wake(self, value: float) -> None:
        self.__contrib_sigma_delta_Clcorr_wake__ = value        
        
    # Declaration of get property contrib_eb_delta_Clcorr_kis for contribution of eb to Clcorr delta.
    @property
    def contrib_eb_delta_Clcorr_kis(self) -> float:
        return self.__contrib_eb_delta_Clcorr_kis__
    
    # Declaration of set property contrib_eb_delta_Clcorr_kis for contribution of eb to Clcorr delta.
    @contrib_eb_delta_Clcorr_kis.setter
    def contrib_eb_delta_Clcorr_kis(self, value: float) -> None:
        self.__contrib_eb_delta_Clcorr_kis__ = value        
        
    # Declaration of get property contrib_eb_delta_Clcorr_wake for contribution of eb to Clcorr delta.
    @property
    def contrib_eb_delta_Clcorr_wake(self) -> float:
        return self.__contrib_eb_delta_Clcorr_wake__
    
    # Declaration of set property contrib_eb_delta_Clcorr_wake for contribution of eb to Clcorr delta.
    @contrib_eb_delta_Clcorr_wake.setter
    def contrib_eb_delta_Clcorr_wake(self, value: float) -> None:
        self.__contrib_eb_delta_Clcorr_wake__ = value                

    # Declaration of get property delta_Cdcorr_kis for corrected drag coefficient delta.
    @property
    def delta_Cdcorr_kis(self) -> float:
        return self.__delta_Cdcorr_kis__

    # Declaration of set property delta_Cdcorr_kis for corrected drag coefficient delta.
    @delta_Cdcorr_kis.setter
    def delta_Cdcorr_kis(self, value: float) -> None:
        self.__delta_Cdcorr_kis__ = value
        
    # Declaration of get property delta_Cdcorr_wake for corrected drag coefficient delta.
    @property
    def delta_Cdcorr_wake(self) -> float:
        return self.__delta_Cdcorr_wake__

    # Declaration of set property delta_Cdcorr_wake for corrected drag coefficient delta.
    @delta_Cdcorr_wake.setter
    def delta_Cdcorr_wake(self, value: float) -> None:
        self.__delta_Cdcorr_wake__ = value        
        
    # Declaration of get property contrib_Cd_delta_Cdcorr_kis for contribution of Cd to Cdcorr delta.
    @property
    def contrib_Cd_delta_Cdcorr_kis(self) -> float:
        return self.__contrib_Cd_delta_Cdcorr_kis__
    
    # Declaration of set property contrib_Cd_delta_Cdcorr_kis for contribution of Cd to Cdcorr delta.
    @contrib_Cd_delta_Cdcorr_kis.setter
    def contrib_Cd_delta_Cdcorr_kis(self, value: float) -> None:
        self.__contrib_Cd_delta_Cdcorr_kis__ = value
        
    # Declaration of get property contrib_Cd_delta_Cdcorr_wake for contribution of Cd to Cdcorr delta.
    @property
    def contrib_Cd_delta_Cdcorr_wake(self) -> float:
        return self.__contrib_Cd_delta_Cdcorr_wake__
    
    # Declaration of set property contrib_Cd_delta_Cdcorr_wake for contribution of Cd to Cdcorr delta.
    @contrib_Cd_delta_Cdcorr_wake.setter
    def contrib_Cd_delta_Cdcorr_wake(self, value: float) -> None:
        self.__contrib_Cd_delta_Cdcorr_wake__ = value        
        
    # Declaration of get property contrib_esb_delta_Cdcorr_kis for contribution of esb to Cdcorr delta.
    @property
    def contrib_esb_delta_Cdcorr_kis(self) -> float:
        return self.__contrib_esb_delta_Cdcorr_kis__
    
    # Declaration of set property contrib_esb_delta_Cdcorr_kis for contribution of esb to Cdcorr delta.
    @contrib_esb_delta_Cdcorr_kis.setter
    def contrib_esb_delta_Cdcorr_kis(self, value: float) -> None:
        self.__contrib_esb_delta_Cdcorr_kis__ = value
        
    # Declaration of get property contrib_esb_delta_Cdcorr_wake for contribution of esb to Cdcorr delta.
    @property
    def contrib_esb_delta_Cdcorr_wake(self) -> float:
        return self.__contrib_esb_delta_Cdcorr_wake__
    
    # Declaration of set property contrib_esb_delta_Cdcorr_wake for contribution of esb to Cdcorr delta.
    @contrib_esb_delta_Cdcorr_wake.setter
    def contrib_esb_delta_Cdcorr_wake(self, value: float) -> None:
        self.__contrib_esb_delta_Cdcorr_wake__ = value        
        
    # Declaration of get property contrib_eb_delta_Cdcorr_kis for contribution of eb to Cdcorr delta.
    @property
    def contrib_eb_delta_Cdcorr_kis(self) -> float:
        return self.__contrib_eb_delta_Cdcorr_kis__
    
    # Declaration of set property contrib_eb_delta_Cdcorr_kis for contribution of eb to Cdcorr delta.
    @contrib_eb_delta_Cdcorr_kis.setter
    def contrib_eb_delta_Cdcorr_kis(self, value: float) -> None:
        self.__contrib_eb_delta_Cdcorr_kis__ = value
        
    # Declaration of get property contrib_eb_delta_Cdcorr_wake for contribution of eb to Cdcorr delta.
    @property
    def contrib_eb_delta_Cdcorr_wake(self) -> float:
        return self.__contrib_eb_delta_Cdcorr_wake__
    
    # Declaration of set property contrib_eb_delta_Cdcorr_wake for contribution of eb to Cdcorr delta.
    @contrib_eb_delta_Cdcorr_wake.setter
    def contrib_eb_delta_Cdcorr_wake(self, value: float) -> None:
        self.__contrib_eb_delta_Cdcorr_wake__ = value
        
    # Declaration of get property delta_Cp for Cp delta.
    @property
    def delta_Cp(self) -> float:
        return self.__delta_Cp__
    
    # Declaration of set property delta_Cp for Cp delta.
    @delta_Cp.setter
    def delta_Cp(self, value: float) -> None:
        self.__delta_Cp__ = value
        
    # Declaration of get property contrib_p_delta_Cp for contribution of p to Cp delta.
    @property
    def contrib_p_delta_Cp(self) -> float:
        return self.__contrib_p_delta_Cp__
    
    # Declaration of set property contrib_p_delta_Cp for contribution of p to Cp delta.
    @contrib_p_delta_Cp.setter
    def contrib_p_delta_Cp(self, value: float) -> None:
        self.__contrib_p_delta_Cp__ = value    
        
    # Declaration of get property contrib_q_delta_Cp for contribution of q to Cp delta.
    @property
    def contrib_q_delta_Cp(self) -> float:
        return self.__contrib_q_delta_Cp__
    
    # Declaration of set property contrib_q_delta_Cp for contribution of q to Cp delta.
    @contrib_q_delta_Cp.setter
    def contrib_q_delta_Cp(self, value: float) -> None:
        self.__contrib_q_delta_Cp__ = value

    # Initialization of function.
    def __init__(self, NVFx=5, NVFy=5, NVFz=10, delta_l=1.4e-3, delta_s=0.7e-3, delta_p=7.5e-2):
        """Initializer of data class uncertainty_intervals.

        -**parameters**

        :param NVFx: N/V scale of the piezoelectric's x-axis.
        :param NVFy: N/V scale of the piezoelectric's y-axis.
        :param NVFz: N/V scale of the piezoelectric's z-axis.
        :param delta_l: uncertainty interval corresponding to the characteristic length.
        :param delta_s: uncertainty interval corresponding to the characteristic span.
        :param delta_p: uncertainty interval corresponding to pressure scanner measurements.

        """

        #####
        ##### Setting attributes.
        #####         
        self.delta_Fx *= NVFx/5
        # y-axis delta.
        self.delta_Fy *= NVFy/5
        # z-axis delta.
        self.delta_Fz *= NVFz/10
        ## Setting characteristic dimension deltas.
        # Length delta.
        self.delta_l = delta_l
        # Span delta.
        self.delta_s = delta_s
        # Setting pressure scanner measurement delta.
        self.delta_p = delta_p

######################################################################################################################
####################################################PUBLIC METHODS####################################################
######################################################################################################################
# Public methods are intended to provide general functionalities on data manipulation, such as automated
# file opening, programmatic variable declaration or fast plotting.

# Public method open_list_of_files.
def open_list_of_files(list_of_files, reload=False, nd=True):
    """Opens a list of TDMS files in a provided reading mode.

    A number of TDMS files are meant to be passed as a list-like argument, together with a custom enum variable
    determining the reading mode.

    - **parameters**, **returns**, **return types**::

    :param list_of_files: list of instances of the fileData object providing TDMS file metadata.
    :param global_dict: dictionary-typed variable within which the instantiated TDMS files are to be programmatically
    allocated. This variable is intended to match with the globals() dictionary statically declared at runtime.
    :param reload: boolean type variable that indicates whether a reload of the files is needed. If True, then the serialized pickle files are ignored, and the TDMS files are loaded.
    :returns: a list of tdmsFileRoot objects representing the instantiations of the provided TDMS files.
    :rtypes: tdmsFileRootlist().

    """
    
    # Getting the main module's frame by employing sys-related built-in methods.
    caller_frame = sys._getframe(0)
    # This conditional stops when the main module ('<module>') is found; if the current frame is not the main frame, then the previous frame
    # is retrieved.
    while caller_frame.f_code.co_name != '<module>':
        caller_frame = caller_frame.f_back

    # Deleting any previously declared variable of the TdmsFileRoot type.
    if "list_of_file_variables" in caller_frame.f_globals.keys():
        globals_deleter([glob for glob in caller_frame.f_globals if (glob in caller_frame.f_globals["list_of_file_variables"])])
        # Deleting the list_of_file_variables to avoid having extra files to open due to previously introduced file paths on that variable.
        globals_deleter(["list_of_file_variables"])

    # Re-declaring the 'list_of_file_variables' variable.
    list_of_file_variables = list()

    # A "for" loop running over the files on the list_of_file_variables variable and programmatically declaring them.
    for file in list_of_files:
        # Printing statement for notifying the beginning of a file's opening process.
        print(colored('----------Attempting ', color='grey', attrs=['bold']) + colored(file.file_alias, color='blue', attrs=['bold']) + colored('----------', color='grey', attrs=['bold']))
        # Conditional for checking whether the type of file to be opened is a TDMS file.
        if type(file) == TdmsFileData:
            # Getting the file path.
            path = file.file_path
            # Replacing possible whitespaces in the file alias.
            file_alias_no_whitespace = file.file_alias.replace(" ", "_")
            # Getting the file path's directory.
            parent_path = "/".join(path.split("/")[:-1])
            # Getting the name of the file to be opened.
            child_path = file.file_alias
            # Getting the files on the file path's directory.
            parent_path_files = os.listdir(parent_path)
            # Getting the extensions on the file path's directory.
            parent_path_file_extensions = [file.split(".")[-1] for file in parent_path_files]
            # Conditional for checking wheter a reload condition has been set or there does not exist a Pickle file in the directory.
            if reload or 'p' not in parent_path_file_extensions:
                # Instantiating a TdmsFileRoot object with the input data.
                caller_frame.f_globals[file_alias_no_whitespace] = TdmsFileRoot(file.file_path, file.file_read_mode, file.file_alias, file.file_ref_mag, nd)
                # Declaring the Pickle file's name as the current file's name and the extension '.p'.
                pickle_file = child_path + '.p'                
                # Building the Pickle file's path as the current file's directory path and the Pickle file's name.
                pickle_file_path = parent_path + "/" + pickle_file
                # Print statement for notifying the saving process of the serialized (Pickled) file.
                print("Saving serialized object in Pickle file: " + colored(pickle_file, color="magenta", attrs=["bold"]))
                # Dumping statement that serializes the file.
                pickle.dump(caller_frame.f_globals[file_alias_no_whitespace], open(pickle_file_path, "wb"))
            # Conditional block hit when a Pickle file is found.
            else:
                # Getting the Pickle file name.
                pickle_file = parent_path_files[parent_path_file_extensions.index('p')]
                # Getting the Pickle file path.
                pickle_file_path = parent_path + "/" + pickle_file
                # Print statement for notifying that a Pickle file has been found and the loading is done from therein.
                print("Pickle file found: " + colored(pickle_file, color="magenta", attrs=["bold"]) + ". Loading data from serialized object.")
                # Assignment of the correspondent global variable to the loading of the Pickle file.
                caller_frame.f_globals[file_alias_no_whitespace] = pickle.load(open(pickle_file_path, "rb"))
            # Printing statement for monitoring the programmatic declaration/instantiation process.
            print("TDMS file " + str(list_of_files.index(file) + 1) + " of " + str(len(list_of_files)) + " opened in " + colored(file.file_read_mode.name, color="yellow", attrs=["bold"]) + " mode; " + colored(file_alias_no_whitespace, color="blue", attrs=["bold"]) + " global variable has been " + colored("created", color="green", attrs=["bold"]) + ".")
            # Appending the alias to the 'list_of_file_variables' list.
            list_of_file_variables.append(file_alias_no_whitespace)
        # Conditional block hit when the type of file is an 'XFoilFileData' type.
        elif type(file) == XFoilFileData:
            # Assignment of 'angles', 'lift', 'drag', 'eff' and 'pitch' variables to the loading of the correspondent .txt file.
            angles, lift, drag, eff, pitch = np.loadtxt(file.filePath, delimiter='\t', usecols=(0, 1, 2, 3, 4), unpack=True)
            # Assignment of the correspondent global variable to an instantiation of the 'XFoilRoot' object.
            caller_frame.f_globals[file.file_alias.replace(" ", "_")] = XFoilRoot(angles, lift, drag, eff, pitch)
            # Appending the alias to the 'list_of_file_variables' list.
            list_of_file_variables.append(file.file_alias.replace(" ", "_"))
            # Printing statement for monitoring the programmatic declaration/instantiation process.
            print("XFoil file " + str(list_of_files.index(file) + 1) + " of " + str(len(list_of_files)) + " opened; " + colored(file.fileAlias.replace(" ", "_"), color="blue", attrs=["bold"]) + " global variable has been " + colored("created", color="green", attrs=["bold"]) + ".")        

    # Damping the list_of_file_variables variable to the global_dict dictionary.
    # global_dict["list_of_file_variables"] = list_of_file_variables
    caller_frame.f_globals["list_of_file_variables"] = list_of_file_variables

# Public method globals_deleter.
def globals_deleter(look_up_list):
    """Deletes entries from a dictionary according to a given list.
    
    This method is intended to update the globals() dictionary statically declared at runtime.

    - **parameters**::

    :param look_up_list: list from which the entries to be deleted are retrieved.
    :param global_dict: dictionary-typed variable within which the instantiated TDMS files are to be programmatically
    allocated. This variable is intended to match with the globals() dictionary statically declared at runtime.
    
    """

    # Getting the main module's frame by employing sys-related built-in methods.
    caller_frame = sys._getframe(0)
    # This conditional stops when the main module ('<module>') is found; if the current frame is not the main frame, then the previous frame
    # is retrieved.
    while caller_frame.f_code.co_name != '<module>':
        caller_frame = caller_frame.f_back

    # A "for" loop that runs over the string literals on the look_up_list variables and deletes them from the global
    # dict dictionary, if found.    
    for glob in look_up_list:
        if glob in caller_frame.f_globals.keys():
            # Printing statement for monitoring the deleting process.
            print(colored(glob, color="blue", attrs=["bold"]) + " global variable has been " + colored("deleted", color="red", attrs=["bold"]) + ".")
            del caller_frame.f_globals[glob]

######################################################################################################################
#####################################################END OF FILE######################################################
######################################################################################################################

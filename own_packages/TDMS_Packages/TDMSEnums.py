"""
Created on Fri Nov 2 15:22:35 2018

@author: azarketa

This file is intended to act as a side-module to scripts that aim at processing TDMS files. It includes the necessary custom enum classes
to provide input information to the classes that process the TDMS files that store MGEP's wind tunnel data.
"""

######################################################################################################################
########################################################PACKAGES######################################################
######################################################################################################################

# enum package necessary to declare custom enumerator classes.
import enum

######################################################################################################################
########################################################CLASSES#######################################################
######################################################################################################################

# Public class tdmsFilereadMode.
class tdmsFilereadMode(enum.Enum):
    """This class provides options for allowing different reading modes of TDMS files."""

    standard = 1
    means = 2
    means_only = 3
    projected = 4
    projected_means_only = 5
    wake_rake = 6
    cobra = 7
    projected_and_wake_rake = 8
    surface_pressure = 9

# Public class deviceAxis.
class deviceAxis(enum.Enum):
    """This class provides options for defining axis information about devices."""

    x = 1
    y = 2
    z = 3
    angle = 4

# Public class runsPerParamMode.
class runsPerParamMode(enum.Enum):
    """This class provides options for setting the information pertaining the group to be added when performing a
    classification of a TDMS file data according to the positional information."""

    kistler = 1
    wake_rake = 2
    cobra = 3
    surface_pressure = 4
    
# Public class addedGroups.
class addedGroups(enum.Enum):
    """This class provides options for setting the information pertaining the group from which a hierarchical
    classification of the positional information is meant to be carried out. As such, it contains the groups that 
    own, by post-processing design, a hierarchical positional information pyramid."""
    
    kistler_group = 2
    wake_rake_group = 3
    cobra_group = 4
    surface_pressure_group = 5
    
# Public class measurands.
class measurands(enum.Enum):
    '''This class provides options for the potential available measurands on a test.'''
    
    length = ["delta_l"]
    span = ["delta_s"]
    area = ["delta_A"]
    temp = ["delta_T"]
    rh = ["delta_RH"]
    barpres = ["delta_P"]
    vel = ["delta_U"]
    velcorr = ["delta_U_corr", "contrib_U_delta_U_corr", "contrib_T_delta_U_corr", "contrib_P_delta_U_corr"]
    dens = ["delta_rho", "contrib_T_delta_rho", "contrib_P_delta_rho", "contrib_RH_delta_rho"]
    visc = ["delta_mu", "contrib_T_delta_mu", "contrib_P_delta_mu", "contrib_RH_delta_mu"]
    Re = ["delta_Re", "contrib_rho_delta_Re", "contrib_U_corr_delta_Re", "contrib_c_delta_Re", "contrib_mu_delta_Re"]
    dynpres = ["delta_q", "contrib_rho_delta_q", "contrib_U_corr_delta_q"]
    Fx =[ "delta_Fx"]
    Fy = ["delta_Fy"]
    Fz = ["delta_Fz"]
    Cl = ["delta_Cl","contrib_Fy_delta_Cl", "contrib_Fx_delta_Cl", "contrib_A_delta_Cl", "contrib_q_delta_Cl"]
    Clcorr_kis = ["delta_Clcorr_kis", "contrib_Cl_delta_Clcorr_kis", "contrib_sigma_delta_Clcorr_kis", "contrib_eb_delta_Clcorr_kis"]
    Clcorr_wake = ["delta_Clcorr_wake", "contrib_Cl_delta_Clcorr_wake", "contrib_sigma_delta_Clcorr_wake", "contrib_eb_delta_Clcorr_wake"]
    Cd_kis = ["delta_Cd_kis", "contrib_Fy_delta_Cd", "contrib_Fx_delta_Cd", "contrib_A_delta_Cd", "contrib_q_delta_Cd"]
    Cd_wake = ["delta_Cd_wake", "contrib_p_delta_Cd", "contrib_q_wake_delta_Cd", "contrib_filt_delta_Cd"]
    Cdcorr_kis = ["delta_Cdcorr_kis", "contrib_Cd_delta_Cdcorr_kis", "contrib_esb_delta_Cdcorr_kis", "contrib_eb_delta_Cdcorr_kis"]
    Cdcorr_wake = ["delta_Cdcorr_wake", "contrib_Cd_delta_Cdcorr_wake", "contrib_esb_delta_Cdcorr_wake", "contrib_eb_delta_Cdcorr_wake"]
    Cp = ["delta_Cp", "contrib_p_delta_Cp", "contrib_q_delta_Cp"]
    
# Public class pressure_taps.
class pressure_taps(enum.Enum):
    '''This class provides the pressure tap position for certain typical configurations.'''
    
    NACA0021_elec = [0,
                     1.097,
                     3.699,
                     7.005,
                     10.617,
                     14.387,
                     18.247,
                     22.162,
                     26.112,
                     30.084,
                     34.071,
                     38.065,
                     42.063,
                     46.063,
                     50.063,
                     54.061,
                     58.059,
                     62.05,
                     66.038,
                     70.022,
                     74.001,
                     77.976,
                     81.946,
                     85.912,
                     89.874,
                     93.832,
                     97.785,
                     101.734,
                     105.679,
                     109.62
                     ]

######################################################################################################################
#####################################################END OF FILE######################################################
######################################################################################################################

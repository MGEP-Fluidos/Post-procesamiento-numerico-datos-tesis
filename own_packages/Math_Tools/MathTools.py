"""
Created on Fri Apr 12 01:35:10 2019

@author: azarketa
This file is intended to act as a side-module for scripts that aim at processing TDMS files. It includes the mathematical tools that are
necessary when processing TDMS files that store MGEP's wind tunnel data.
"""

######################################################################################################################
########################################################PACKAGES######################################################
######################################################################################################################

# numpy and scipy are intended to perform numeric calculations.
import numpy as np

# importing scipy.
import scipy as sc

# scipy.signal is intended to perform signal processing operations, such as filtering.
import scipy.signal as scsg

# TDMS_Packages.TDMSClasses is a custom package containing custom class objects used on the workflow herein.
# import TDMS_Packages.TDMSClasses as TDMSC

# TDMS_Packages.TDMSEnums is a custom package containing custom Enum objects used on the workflow herein.
# import TDMS_Packages.TDMSEnums as TDMSE

# termcolor.color to perform colored terminal prints.
from termcolor import colored

# Importing matplotlib
import matplotlib as mpl

######################################################################################################################
######################################################FUNCTIONS#######################################################
######################################################################################################################

# get_pdf() function.
def get_pdf(data, bins=100):
    '''Obtains the normalized PDF of the input.
    
    - **parameters**, **return**, **return types**::
    
        :param data: data from which to extract the PDF.
        :param bins: number of bins to be included in the PDF. Default is 50.
        :return: array containing the PDF.
        :rtype: list.
    
    '''
    
    dy = (max(data) - min(data))/bins
    dys = [min(data) + dy*b for b in np.arange(1, bins)]

    dist_values = list()
    data_length = len(data)

    for e, dy in enumerate(dys[1:]):
        e += 1
        dist_values.append(len([y for y in data if y<dy and y>dys[e - 1]])/data_length)
        
    return np.array([100*i/bins for i in range(1, bins-1)]), np.array(dist_values)

# get_cdf() function.
def get_cdf(data):
    '''Obtains the normalized cummulative distribution function of the input.
    
    - **parameters**, **return**, **return types**::

        :param data: data from which to extract the cdf.         
        :return: list containing the cdf, array containing the linear space on the interval [0, 1] with a number of points equal
        to the length of the input argument 'data'.
        :rtype: list, array
    '''
    
    # Return statement computing the normalized cdf (by considering the minimum and maximum values of data through the 'sorted()'
    # function), and the linear space on the interval [0, 1].
    return [(b - sorted(data)[0])/(sorted(data)[-1] - sorted(data)[0]) for b in sorted(data)], np.linspace(0, 1, len(data))
        
#  butterworth_filter() function.
def butterworth_filter(data, sampling_freq = 850, filter_order = 2, cut_off_freq = 20, passband='low'):    
    '''Applies a Butterworth filter customized by input parameters on the input data.
    
    - **parameters**, **return**, **return type**::
    
    :param data: a list- or array-like object to apply the filter upon.
    :param sampling_freq: an integer value providing the frequency at which the data is sampled.
    :param filter_order: an integer value that determines the order of the Butterworth filter to be applied.
    :param cut_off_freq: an integer value that determines the cut-off frequency of the Butterworth filter to be applied.
    :param passband: either 'high' or 'low', depending on the frequencies to filter.
    :return: an array representing the filtered data.
    :rtype: ndarray.
    
    '''
    
    # Normalize the frequency.
    normalized_freq = cut_off_freq / (sampling_freq / 2.0)
    
    if passband == 'low':
        # Getting Butterworth filter constants for low-pass filter.
        b, a = scsg.butter(filter_order, normalized_freq, 'low')
    else:
        # Getting Butterworth filter constants for high-pass filter.
        b, a = scsg.butter(filter_order, normalized_freq, 'high')

    # Return statement.
    return scsg.filtfilt(b, a, data)

# deficit_curve_processor() function.
def deficit_curve_processor(data, threshold=0.1):
    '''Processes a deficit curve based on the maximum-point approach.
    
    The maximum-point approach is an in-house developed post-processing approach that computes the deficit width by first identifying the 
    maximum deficit point. In contrast to looking for the wake beginning by starting the survey from the tails, the maximum-point approach 
    ensures that the starting-point lies within the wake itself. Thus, it avoids the uncertainties associated with erratic fluctuations of 
    the deficit outside the wake, where noise-originated peaks may cause a mismatch on the identification of the wake.
    
    - **parameters**, **returns**, **return types**:
        
    :param data: a list- or array-like object containing the deficit curve data.
    :param threshold: a normalized value (0, 1) that represents the fraction of the maximum value of the deficit curve beyond which the points
    of the curve are consiered null. For filtering purposes. Default is 0.1.
    :returns: filtered data according to the deficit-curve smoothing carried out by the maximum-point approach;
    standard deviation of discarded data points;
    min/max indices of the filtered deficit curve where the wake starts/ends.
    :rtypes: list, float, tuple(int, int).
    
    '''
    
    # Getting the maximum deficit point's index.
    max_index = data.index(np.max(data))
    
    # Declaring 'highest_min_index' and 'lowest_min_index' to store the values of the indices
    # beyond which the values of the deficit are assumed to be 0 (farfield zone).
    highest_min_index = None
    lowest_min_index = None
    
    # Assigning the 'max_index' value to an 'index' variable before starting the while loop
    # that will identify the high index for which the deficit curve will be truncated.    
    index = max_index
    # Notice that the truncation criterion is set to a tenth of the maximum value.
    while data[index] > (threshold*data[max_index]):
        index += 1
        if index == len(data) - 1:
            break
    # Assigning the 'index' vlaue to the 'highest_min_index' variable.
    highest_min_index = index
    
    # Assigning the 'max_index' value to an 'index' variable before starting the while loop
    # that will identify the low index for which the deficit curve will be truncated.    
    index = max_index
    # Notice that the truncation criterion is set to a tenth of the maximum value.
    while data[index] > (data[max_index] / 10):
        index -= 1
        if index == 0:
            break
    # Assigning the 'index' vlaue to the 'lowest_min_index' variable.    
    lowest_min_index = index

    # Getting values of discarded data points.
    discarded_data = [data[index] for index in np.arange(0, lowest_min_index)] + [data[index] for index in np.arange(highest_min_index, len(data))]
    # Computing standard deviation of discarded data points.
    discarded_std = np.std(discarded_data)

    # Building filtered deficit curve.
    filtered_data = [0 for _ in np.arange(0, lowest_min_index)] + [data[index] for index in np.arange(lowest_min_index, highest_min_index)] + [0 for _ in np.arange(highest_min_index, len(data))]

    # Return statement.
    return filtered_data, discarded_std, (lowest_min_index, highest_min_index)

# least_squares() function.
def least_squares(x: list(), y: list()):
    '''Function performing a least squares fit on the provided data by means of the np.polyfit() method.
    
    - **parameters**, **return**, **return type**:
    
    :param x: list-like structure providing the x-axis data array for the fitting.
    :param y: list-like structure providing the y-axis data array for the fitting.
    :return: slope and 0-ordinate of the fitted data (m, y0), and the variance estimates of those coefficients (rss).
    :rtype: float, float, ndarray.
    
    '''
    
    # Calling np.polyfit() method with deg=1 and flagging cov to 'True' in order to get variance estimates.
    coeffs, cov = np.polyfit(x, y, deg=1, full=False, cov=True)
    
    # Assigning slope and 0-ordinate values to 'm' and 'y0' variables.
    m, y0 = coeffs[0], coeffs[1]
    yfit = m*x + y0
    rss = (1/(len(x) - 1))*sum((yfit - y)**2)
    # Assigning the diagonal of the covariance matrix to 'rss' variable.
#     rss = np.sqrt(np.sum([_[i]**2 for i, _ in enumerate(cov)]))
#     rss = cov[0][0]
    
    # Return statement.
    return m, y0, rss

# derivative() function.
def derivative(y, h, order=1):
    '''Returns the 1st or 2nd order derivative of an input function, depending on the value of 'order' [1, 2].
    
    The derivatives are calculated using forward and backward schemes for the extreme points, whereas central differencing
    is used for the points in between.
    
    -**parameters**, **return**, **return type**:
    
    :param y: the function to be derivated.
    :param h: the x-axis distance between points, or the spacing.
    :param order: the order of the derivative, either 1 or 2. Default is 1.
    :return: derived array dy.
    :rtype: ndarray.
    
    '''
    
    # Asserting that the provided input function is either a list or a numpy array.
    assert type(y) in [type(list()), type(np.array([0]))], "Provide either a list or an array as an input function to derive."
    
    # Asserting that the provided derivative order has a correct value.
    assert order in [1, 2], "Provide a valid derivative order, [1, 2]."
    
    # Conditional for checking whether the input function is a numpy array; if it is a list, then convert to array.
    if type(y) != type(np.array([0])):
        y = np.array(y)
    
    # Assigning a zero-populated array to 'dy' variable, which intends to store the derivative function.
    dy = np.zeros(len(y))
    ## Conditional for checking the order of the derivative to compute.
    # First order derivative.
    if order == 1:
        # Forward differencing for left-hand side extreme.
        dy[0] = -3*y[0] + 4*y[1] - y[2]
        # Central differencing.
        dy[1:-1] = y[2:] - y[:-2]
        # Backard differencing for right-hand side extreme.
        dy[-1] = 3*y[-1] - 4*y[-2] + y[-3]
        # Dividing by two times the spacing.
        dy /= 2*h
    # Second order derivative.
    elif order == 2:
        # Forward differencing for left-hand side extreme.
        dy[0] = 2*y[0] - 5*y[1] + 4*y[2] - y[3]
        # Central differencing.
        dy[1:-1] = y[2:] - 2*y[1:-1] + y[:-2]
        # Backard differencing for right-hand side extreme.
        dy[-1] = 2*y[-1] - 5*y[-2] + 4*y[-3] - y[-4]
        # Dividing by two times the spacing.
        dy /= h**2
        
    # Return statement.
    return dy

# zero_runs() function.
def max_range_in_array(x, val, mode='equal'):
    '''Given an array 'x' and a value 'val', computes the maximum succesive range within the array for which a given criterion is hold
    regarding 'val'.
    
    The criterion is dictated by the input parameter 'mode', which may be either 'equal' or 'less'. If equal, the function looks for
    the maximum succesive range for which the members of the array match the value 'val'. If 'less', the function looks for the maximum
    succesive range for which the members of the array lie below the value 'val'.
    
    -**parameters**, **return**, **rtype**
    
    :param x: array for which the maximum range is sought.
    :param val: matching/threshold value.
    :param mode: either 'equal' or 'less', determines the maximum succesive range finding method. Default is 'equal'.
    :return: start index of maximum succesive range, length of maximum succesive range.
    :rtype: tuple(int, int).
    
    '''
    
    # Asserting that the provided 'mode' parameter has an admissible value.
    assert mode in ['equal', 'less'], "Provide a valid 'mode' parameter, either 'equal' or 'less'."
    
    ## Obtaining boolean array with 'True' values where elements match the input criterion.
    # Boolean array if mode='equal'.
    if mode == 'equal':
        ismatch = np.concatenate(([0], np.equal(x, val).view(np.int8), [0]))
    # Boolean array if mode='less'.
    else:
        ismatch = np.concatenate(([0], np.less(x, val).view(np.int8), [0]))
        
    # Obtaining absolute differences for getting the maximum succesive range.
    absdiff = np.abs(np.diff(ismatch))
    # Getting ranges where the 'absdiff' variable is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    # Assigning 0 value to 'maxrange' variable, which intends to store the maximum succesive range length.
    maxrange = 0
    # Assigning [0, len(x)] to 'maxrang' variable, which intends to store the maximum succesive range start/end indices.
    maxrang = list([0, len(x)])
    # Looping over the found ranges for getting the maximum range.
    for e, rang in enumerate(ranges):        
        maxrange, maxrang = (np.diff(rang)[0], rang) if np.diff(rang)[0] > maxrange else (maxrange, maxrang)
    # Return statement.
    return maxrang[0], maxrange

# FFT() function.
def FFT(xt, yt):
    '''Computes unit-coherent FFT of an input signal.
    
    Here, unit-coherent means that, first of all, the input is assumed to be given in a sec-unit temporal
    basis (i.e. the x vector is a temporal vector representing seconds); secondly, it means that the
    output coming from the built-in Python function np.fft.fft() is scaled so that the obtained amplitudes
    correspond to the physical amplitudes present on the temporal signal.
    
    - **parameters**, **returns**, **return types**
    
    :param xt: ndarray corresponding to input temporal vector.
    :param yt: ndarray corresponding to input signal vector.
    :returns: xfft, the frequencies corresponding to the fft-ed signal's frequency domain; yfft, the fft-ed
    signal; the period T of the signal. Notice that xfft and yfft are not of the same length, with xfft
    just comprising the positive frequency terms.
    :rtypes: ndarray, ndarray, float.
    
    '''
    
    # Computing period of linear space as the difference between initial and final points.
    T = xt[-1] - xt[0]
    # Computing the FFT components.
    ## xfft are the frequency components of the spectrum. Notice that:
    ##      - the vector length passed to the function (n) equals the half length of the xs vector (or the
    ## x-axis length of the rectangular pulse vector).
    ##      - the sample spacing (d) needs to be rescaled according to the considered period (T). The 'true'
    ## sample spacing equals the distance between two consecutive points of the xs vector (xs[1] - xs[0]).
    ##      - just the first half of the frequencies is considered (positive frequencies).
    xfft = np.fft.fftfreq(n=len(xt//2), d=(xt[1]-xt[0]))[:len(xt)//2]
    ## yfft are the Fourier transform components of the spectrum. Notice that:
    ##      - the call to the actual fft method is preceded by a normalizing factor, which equals the
    ## inverse of the frequency vector length.
    yfft = (1/len(xfft))*np.fft.fft(a=yt)
    # Return statement.
    return xfft, yfft, T

# IFFT() function.
def IFFT(xfreq, yfreq):
    '''Computes unit-coherent IFFT of an input FFT signal.
    
    Here, unit-coherent means that, first of all, the input is assumed to be given in a frequency-unit
    FFT form (i.e. the x vector is a frequency vector representing the x-axis of a FFT signal); secondly,
    it means that the output coming from the built-in Python function np.fft.ifft() is scaled so that the
    obtained temporal signal corresponds to the physical amplitudes present on the FFT signal.
    
    - **parameters**, **returns**, **return types**
    
    :param xfreq: ndarray corresponding to input frequency vector.
    :param yfreq: ndarray corresponding to input amplitud-phase vector.
    :returns: xt, the temporal vector corresponding to the ifft-ed signal's temporal domain; yt, the
    ifft-ed signal; the frequency lapse F of the FFT signal.
    :rtypes: ndarray, ndarray, float.
    
    '''
    
    # Computing the frequency lapse of the signal.
    F = xfreq[-1] - xfreq[0]
    # Obtaining the time-step of the temporal signal.
    dt = 1/((xfreq[2] - xfreq[1])*len(yfreq))
    # Computing the temporal signal.
    xt = np.array([i*dt for i in range(0, len(yfreq))])
    # Computing the inverse FFT of the input signal.
    yt = len(xfreq)*np.fft.ifft(yfreq)
    # Return statement.
    return xt, yt, F

# signal_analysis() function.
def signal_analysis(xs, ys, rtype='PSD', nperseg=2, print_energies=False):
    '''Provides, given an input function, its FFT, PSD and/or autocorrelation signals.
    
    - **parameters**, **return**
    
    :param xs: ndarray representing the temporal vector of the input signal, measured in secs.
    :param ys: ndarray representing the amplitude vector of the input signal.
    :param rtype: optional string, either 'FFT', 'PSD', 'AUTOCORR' or 'FULL', providing info about the
    processing mode of the SIGNAL. Default is 'PSD', meaning that the power spectrum of the signal is
    returned.
    :param nperseg: power of 2 by which the length of the input signal is to be divided in order
    to obtain the number of points per segment for the Welch method. Default is 2.
    :returns: if 'ALL', then FFT, PSD and autocorrelation signals are returned; otherwise, the value of the
    'rtype' parameter indicates which processed signal is returned.
    :rtypes: tuple(array, array), tuple(array, array), tuple(array, array).
    
    '''
    
    # Assert statement for ensuring that the 'PSDMode' input variable has a proper value.
    assert rtype in ['FFT', 'PSD', 'AUTOCORR', 'FULL'], "Provide a valid 'rtype' value ('FFT'/'PSD'/'AUTOCORR'/'FULL')."
    
    ## Conditional tree for specifying workflow according to value of 'rtype'.
    # If 'rtype' equals 'FFT' or 'FULL', perform FFT operation.
    if rtype == 'FFT' or rtype == 'FULL':
        # Computing the unit-coherent Fourier transform of signal.
        xfft, yfft, T = FFT(xs, ys)
        # Obtaining the modulus of the FFT signal.
        yfftr = np.abs(yfft[:len(xfft)])
        # Conditional to check whether 'rtype' equals 'FFT'; if so, then return FFT-ed signal.
        if rtype == 'FFT':
            # Return statement.
            return (xfft, yfftr)
    # If 'rtype' equals 'PSD' or 'FULL', perform PSD operation.
    elif rtype == 'PSD' or rtype == 'FULL':        
        # Computing the number of signal signal values per segment for the 'welch' method.
        nperseg = 2**(int(np.log2(len(xs))) - nperseg)
        # Computing PSD of the signal by the Welch method.
        xpsd, ypsd = scsg.welch(ys, fs=1/(xs[1] - xs[0]), nperseg=nperseg)
        # Conditional to check whether 'rtype' equals 'PSD'; if so, then return PSD-ed signal.
        if rtype == 'PSD':
            # Return statement.
            return (xpsd, ypsd)
    # If 'rtype' equals 'AUTOCORR' or 'FULL', perform autocorrelation operation.
    elif rtype == 'AUTOCORR' or rtype == 'FULL':
        # Computing autocorrelation in 'full' mode from scipy.signal.fftoconvolve() method.
        autocorr = scsg.fftconvolve(ys, ys[::-1], 'full')
        # Trimming the resultant autocorrelation vector in half.
        autocorr = autocorr[len(autocorr)//2:]
        # Normalizing autocorrelation vector.
        autocorr /= max(autocorr)
        # Conditional to check whether 'rtype' equals 'AUTOCORR'; if so, then return autocorrelation signal.
        if rtype == 'AUTOCORR':
            return (xs, autocorr)

    # If 'rtype' equals 'FULL' then check whether an energetic analysis is to be printed, and return FFT, PSD and
    # autocorrelation signals.
    if rtype == 'FULL':
        if print_energies:
            # Printing statement showing that the obtained results match with theoretical predictions; concretely, it
            # is shown that the energy obtained from the integration of the squared signal matches the energy obtained
            # from the integration of the psd signal.
            print("Energy of the signal, obtained from direct integration: ",
                  sc.integrate.simps(np.real(ys**2), xs),
                  "\nPower of the signal, obtained from direct integration: ",
                  (1/T)*sc.integrate.simps(np.real(ys**2), xs),
                  "\nPower of the signal, obtained from psd integration:",
                  sc.integrate.simps(ypsd, xpsd))
        # Return statement.
        return (xfft, yfftr), (xpsd, ypsd), (xs, autocorr)

# corrector() function.
def corrector(Lift_file, Drag_file):
    '''Applies wind tunnel wall-corrections to a pair of lift-drag curves.
    
    The corresponding correction formulas may be found on either of the following references:
        - Garner H, Rogers E, Acum W, Maskell E. 1966. Subosnic Wind Tunnel Wall Corrections. Teddington, Middlesex, England
        - Selig M, Deters R, Wiliamson G. 2014. Wind Tunnel Testing Airfoils at Low Reynolds Numbers.
    
    - **parameters**::
            
    :param Lift_file: a TdmsFileRoot type object containing a 'kistler_group' group that comprises the correspondent lift data.
    It is assumed that the file contains velocity, AoA and Cm data also.
    :param Drag_file: a TdmsFileRoot type object containing a 'wake_rake_group' group that comprises the correspondent drag data.
    It is assumed that the file contains velocity and AoA data also.
    '''
    
     # Variables for determining if the groups 'means_group' are deleted during the processing, or otherwise need to be deleted.
    lift_means_group_deleted = False
    drag_means_group_deleted = False

    # Conditional tree checking whether the 'Lift_file' variable needs to be processed further to get mean values.    
    if not hasattr(Lift_file.groups_added, "means_group"):
        # If the 'means_group' group is not present, then process the means and set the 'lift_means_group_deleted' variable to 'True'.
        lift_means_group_deleted = True
        Lift_file.process_means()
    else:
        # No action otherwise.
        pass
    
    # Conditional tree checking whether the 'Drag_file' variable needs to be processed further to get mean values.
    if not hasattr(Drag_file.groups_added, "means_group"):
        # If the 'means_group' group is not present, then process the means and set the 'drag_means_group_deleted' variable to 'True'.
        drag_means_group_deleted = True
        Drag_file.process_means()
    else:
        # No action otherwise.
        pass
    
    # Calling internal __get_position_hierarchy__() method for obtaining the positional information of the 'Lift_file' variable.
#     eval_string, present_pos_params = Lift_file.__get_position_hierarchy__(departing_group=TDMSE.addedGroups.kistler_group)
    # Getting the eval string for the parent positional parameter.
    parent_eval_string = ".".join(eval_string.split('.')[:-1]).replace("self", "Lift_file") if len(present_pos_params) > 1 else eval_string.replace("self", "Lift_file")
    
    # Loading uncorrected variables from lift file.
    # Velocity from lift file.
    Vu_lift = Lift_file.groups_added.means_group.means_channels.fix_probe_signal
    # Angle of attack from lift file.
    Au_lift = eval(parent_eval_string + ".angle_pos")
    # Lift coefficient.
    Clu = eval(parent_eval_string + ".lift")
    # Z-axis momentum coefficient.
    Cmu = eval(parent_eval_string + ".corrected_kistler_mz_signal")
    
    # Calling internal __get_position_hierarchy__() method for obtaining the positional information of the 'Drag_file' variable.
#     eval_string, present_pos_params = Drag_file.__get_position_hierarchy__(departing_group=TDMSE.addedGroups.wake_rake_group)
    # Getting the eval string for the parent positional parameter.
    parent_eval_string = ".".join(eval_string.split('.')[:-1]).replace("self", "Drag_file") if len(present_pos_params) > 1 else eval_string.replace("self", "Drag_file")
    
    # Loading uncorrected variables from drag file.
    # Velocity from drag file.
    Vu_drag = Drag_file.groups_added.means_group.means_channels.fix_probe_signal
    # Angle of attack from drag file.
    Au_drag = eval(parent_eval_string + ".angle_pos")
    # Drag coefficient.
    Cdu = eval(parent_eval_string + ".angle_cd")
    
    # Geometrical parameters.
    # Chord.
    c = Lift_file.__ref_mag__.__length__
    # Span.
    s = Lift_file.__ref_mag__.__span__
    # Thickness.
    t = Lift_file.__ref_mag__.__thick__
    # Wind tunnel height equipped with end-plates.
    H = 0.902
    # Wind tunnel span.
    S = 0.75
    # Wind tunnel's cross-section.
    C = H * S
    # K1 parameter; takes the value 0.52 for a drag item spanning the height of the wind tunnel.
    K1 = 0.52
    # Drag item's volume. For an airfoil it is estimated as 0.7·t·c·s.
    V = 0.7 * t * c * s
    
    # Flow-dependent paramters.
    # Solid blockage.
    sol_bl = (K1 * V) / C**(1.5)
    # Wake blockage.
    wake_bl = [(c / (2 * H)) * Cdu_ for Cdu_ in Cdu]
    # Blockage.
    bl = [sol_bl + wake_bl_ for wake_bl_ in wake_bl]
    # Sigma parameter.
    sigma = (np.pi**2 / 48) * (0.15 / 0.902)**2
    
    # Velocity correction.
    V_lift = [Vu_lift_bl_[0] * (1 + Vu_lift_bl_[1]) for Vu_lift_bl_ in zip(Vu_lift, bl)]
    V_drag = [Vu_drag_bl_[0] * (1 + Vu_drag_bl_[1]) for Vu_drag_bl_ in zip(Vu_drag, bl)]
    # Angle of attack correction.
    A_lift = [Au_lift_Clu_Cmu_[0] + ((57.3 * sigma) / (2 * np.pi)) * (Au_lift_Clu_Cmu_[1] + 4 * Au_lift_Clu_Cmu_[2]) for Au_lift_Clu_Cmu_ in zip(Au_lift, Clu, Cmu)]
    A_drag = [Au_drag_Clu_Cmu_[0] + ((57.3 * sigma) / (2 * np.pi)) * (Au_drag_Clu_Cmu_[1] + 4 * Au_drag_Clu_Cmu_[2]) for Au_drag_Clu_Cmu_ in zip(Au_drag, Clu, Cmu)]
    # Lift coefficient correction.
    Cl = [Clu_bl_[0] * (1 - sigma) / (1 + Clu_bl_[1])**2 for Clu_bl_ in zip(Clu, bl)]
    # Momentum coefficient correction.
    Cm = [Cmu_bl_Cl_[0] * (Cmu_bl_Cl_[2] * sigma * 0.25 * (1 - sigma)) / (1 + Cmu_bl_Cl_[1])**2 for Cmu_bl_Cl_ in zip(Cmu, bl, Cl)]
    # Drag coefficient correction.
    Cd = [Cdu_bl_[0] * (1 - sol_bl) / (1 + Cdu_bl_[1])**2 for Cdu_bl_ in zip(Cdu, bl)]

    # Sorting variables according to angle before proceeding to the setting process.
    Au_lift, Vu_lift, Clu, Cmu = zip(*sorted(zip(Au_lift, Vu_lift, Clu, Cmu)))
    A_lift, V_lift, Cl, Cmu = zip(*sorted(zip(A_lift, V_lift, Cl, Cm)))
    # Setting the correspondent parameters on the parent positional information parameter's hierarchical position.    
    if hasattr(Lift_file.groups_added, "corrected_group") and (Lift_file.groups_added.corrected_group.reference_file == Drag_file.__path__):
        # If the 'corrected_group' group is present and the reference drag file matches the one being used herein, then the correction process
        # is skipped.
        print(colored("Corrections already done with same reference file; skipping process.", color="green", attrs=["bold"]))
        # Return statement.
        return
    elif hasattr(Lift_file.groups_added, "corrected_group"):
        # If the 'corrected_group' is present but the reference drag file does not match the one being used herein, then new values must be
        # set on the corresponding variables.
        # Setting corrected and uncorrected velocities.
        Lift_file.groups_added.corrected_group.V_cor = V_lift
        Lift_file.groups_added.corrected_group.V_uncor = Vu_lift
        # Setting corrected and uncorrected AoAs.
        Lift_file.groups_added.corrected_group.angle_pos_cor = A_lift
        Lift_file.groups_added.corrected_group.angle_pos_uncor = Au_lift
        # Setting corrected and uncorrected lift coefficients.
        Lift_file.groups_added.corrected_group.lift_cor = Cl
        Lift_file.groups_added.corrected_group.lift_uncor = Clu
        # Setting corrected and uncorrected momentum coefficients.
        Lift_file.groups_added.corrected_group.momentum_cor = Cm
        Lift_file.groups_added.corrected_group.momentum_uncor = Cmu
        # Setting the reference drag file used for the corrections.
        Lift_file.groups_added.corrected_group.reference_file = Drag_file.__path__
    else:
        # If the 'corrected_group' group is not present, then it must be instantiated as a new group and the corresponding attribute hierarchy
        # must be set.
        # Creating the 'corrected_group' group.
        Lift_file.groups_added.__add_isolated_group__("corrected_group")
        # Setting the attributes and values corresponding to corrected and uncorrected velocities.
        Lift_file.groups_added.corrected_group.__setattr__("V_cor", V_lift)
        Lift_file.groups_added.corrected_group.__setattr__("V_uncor", Vu_lift)
        # Setting the attributes and values corresponding to corrected and uncorrected AoAs.
        Lift_file.groups_added.corrected_group.__setattr__("angle_pos_cor", A_lift)
        Lift_file.groups_added.corrected_group.__setattr__("angle_pos_uncor", Au_lift)
        # Setting the attributes and values corresponding to corrected and uncorrected lift coefficients.
        Lift_file.groups_added.corrected_group.__setattr__("lift_cor", Cl)
        Lift_file.groups_added.corrected_group.__setattr__("lift_uncor", Clu)
        # Setting the attributes and values corresponding to corrected and uncorrected moment coefficients.
        Lift_file.groups_added.corrected_group.__setattr__("momentum_cor", Cm)
        Lift_file.groups_added.corrected_group.__setattr__("momentum_uncor", Cmu)
        # Setting the attribute and value corresponding to the drag file used as reference for the corrections.
        Lift_file.groups_added.corrected_group.__setattr__("reference_file", Drag_file.__path__)
        
    # Sorting variables according to angle before proceeding to the setting process.
    Au_drag, Vu_drag, Cdu = zip(*sorted(zip(Au_drag, Vu_drag, Cdu)))
    A_drag, V_drag, Cd = zip(*sorted(zip(A_drag, V_drag, Cd)))
    # Setting the correspondent parameters on the parent positional information parameter's hierarchical position.    
    if hasattr(Drag_file.groups_added, "corrected_group"):
        # If the 'corrected_group' is present but the reference drag file does not match the one being used herein, then new values must be
        # set on the corresponding variables.
        # Setting corrected and uncorrected velocities.
        Drag_file.groups_added.corrected_group.V_cor = V_drag
        Drag_file.groups_added.corrected_group.V_uncor = Vu_drag
        # Setting corrected and uncorrected AoAs.
        Drag_file.groups_added.corrected_group.angle_pos_cor = A_drag
        Drag_file.groups_added.corrected_group.angle_pos_uncor = Au_drag
        # Setting corrected and uncorrected drag coefficients.
        Drag_file.groups_added.corrected_group.drag_cor = Cd
        Drag_file.groups_added.corrected_group.drag_uncor = Cdu
    else:
        # If the 'corrected_group' group is not present, then it must be instantiated as a new group and the corresponding attribute hierarchy
        # must be set.
        # Creating the 'corrected_group' group.
        Drag_file.groups_added.__add_isolated_group__("corrected_group")
        # Setting the attributes and values corresponding to corrected and uncorrected velocities.
        Drag_file.groups_added.corrected_group.__setattr__("V_cor", V_drag)
        Drag_file.groups_added.corrected_group.__setattr__("V_uncor", Vu_drag)
        # Setting the attributes and values corresponding to corrected and uncorrected AoAs.
        Drag_file.groups_added.corrected_group.__setattr__("angle_pos_cor", A_drag)
        Drag_file.groups_added.corrected_group.__setattr__("angle_pos_uncor", Au_drag)
        # Setting the attributes and values corresponding to corrected and uncorrected drag coefficients.
        Drag_file.groups_added.corrected_group.__setattr__("drag_cor", Cd)
        Drag_file.groups_added.corrected_group.__setattr__("drag_uncor", Cdu)
        
    # Conditional statement to check whether the 'means_group' group of the 'Lift_file' variable needs to be hidden.
    if lift_means_group_deleted:
        # Attribute instantiation and consequent deleting in case the "means_group" group needs to be deleted.
        Lift_file.groups_added.__setattr__("__means_group__", Lift_file.groups_added.means_group)
        Lift_file.groups_added.__delattr__("means_group")
    else:
        # No action otherwise.
        pass
    
    # Conditional statement to check whether the 'means_group' group of the 'Lift_file' variable needs to be hidden.
    if drag_means_group_deleted:
        # Attribute instantiation and consequent deleting in case the "means_group" group needs to be deleted.
        Drag_file.groups_added.__setattr__("__means_group__", Drag_file.groups_added.means_group)
        Drag_file.groups_added.__delattr__("means_group")
    else:
        # No action otherwise.
        pass

# fix_probe_correction() function.
def fix_probe_correction(P, T):
    """It computes the necessary correction on the measured velocity in order to account for non-nominal pressure/temperature conditions.

    -**parameters**, **returns**, **return types**

    :param P: non-nominal barometric pressure, in mbar.
    :param T: non-nominal ambient temperature, in ºC.
    :returns: slopes of the temperature- and pressure-dependent corrections; correction in percentage on the measured velocity.
    :rtypes: float, float, float.

    """

    # Nominal ambient temperature, in ºC, provided by the manufacturer. Corresponds to the first abscissa point.
    xT1 = 16
    # Temperature correction for nominal temperature, equals 1. Corresponds to the first ordinate point.
    yT1 = 1
    # Non-nominal ambient temperature, in ºC, for which a computable correction is known. Corresponds to the second abscissa point.
    xT2 = 5
    # Temperature correction for non-nominal temperature. Corresponds to the second ordinate point.
    yT2 = 0.98
    # Nominal barometric pressure, in mbar, provided by the manufacturer. Corresponds to the first abscissa point.
    xP1 = 1013.25
    # Pressure correction for nominal pressure, equals 1. Corresponds to the first ordinate point.
    yP1 = 1
    # Non-nominal barometric pressure, in mbar, for which a computable correction is known. Corresponds to the second abscissa point.
    xP2 = 807
    # Pressure correction for non-nominal pressure. Corresponds to the second ordinate point.
    yP2 = 1.12

    # Computing slope of temperature-dependent relation, from abscissa/ordinate point pairs.
    mT = (yT2 - yT1)/(xT2 - xT1)
    # Computing slope of pressure-dependent relation, from abscissa/ordinate point pairs.
    mP = (yP2 - yP1)/(xP2 - xP1)
    # Obtaining correction in temperature.
    corrT = yT1 + mT*(T - xT1)
    # Obtaining correction in pressure.
    corrP = yP1 + mP*(P - xP1)

    # Return statement.
    return mT, mP, corrT*corrP

# moist_air_density_and_viscosity() function.
def moist_air_density_and_viscosity(p: float, t: float, RH: float, mode: str='noder') -> tuple():
    """Computes the density and viscosity of humid air given the input parameters of barometric pressure,
    ambient temperature and relative humidity, and their respective derivatives if requested.
    
    -**parameters**, **returns**, **return types**
    
    :param p: value of barometric pressure (Pa).
    :param t: value of ambient temperature (ºC).
    :param RH: value of relative humidity (0 <= RH <= 1).
    :param mode: string, either 'noder' (just returns density and viscosity) or 'der' (returns derivatives also).
    Default is 'noder'
    :returns: density (kg/m^{3}) and viscosity (kg/(m·s)) corresponding to state of moist air given the input parameters.
    Density, viscosity and their derivatives (drho_dt, drho_dp, drho_dRH), (dmu_dt, dmu_dp, dmu_dRH) with respect to
    ambient conditions if mode='der'.
    :rtypes: tuple(float, float) (mode='noder')
             tuple(float, float, float, float, float, float, float, float) (mode='der')
    
    """
    
    assert mode=='noder' or mode=='der', 'The "mode" parameter must take either of the following values: "noder", "der".'

    # Changing units from ºC to ºK of input temperature.
    T = t + 273.15
    # Setting molar fraction of CO2 in air (not measured; standard value).
#     xCO2 = 400e-06

    # Constants for the computation of the amount of water vapour xv.
    A = 1.2378847e-05
    B = -1.912136e-02
    C = 33.93711047
    D = -6.3431645e3
    alpha = 1.00062
    beta = 3.14e-08
    gamma = 5.6e-07

    ## Computation of the amount of water vapour xv.
    xv = RH*(alpha + beta*p + gamma*t**2)*np.exp(A*T**2 + B*T + C + D/T)/p
    ## Computation of derivatives of the amount of water vapour xv.
    # Derivative with respect to temperature.
    dxv_dt = (RH/p)*np.exp(A*T**2 + B*T + C + D/T)*(2*gamma*t + (alpha + beta*p + gamma*t**2)*(2*A*T + B - D/T**2))
    # Derivative with respect to pressure.
    dxv_dp = -(RH/p**2)*np.exp(A*T**2 + B*T + C + D/T)*(alpha + gamma*t**2)
    # Derivative with respect to relative humidity.
    dxv_dRH = (alpha + beta*p + gamma*t**2)*np.exp(A*T**2 + B*T + C + D/T)/p

    # Constants for the computation of the compressibility factor.
    a0 = 1.58123e-06
    a1 = -2.9331e-08
    a2 = 1.1043e-10
    b0 = 5.707e-06
    b1 = -2.051e-08
    c0 = 1.9898e-04
    c1 = -2.376e-06
    d = 1.83e-11
    e = -0.765e-08
    ## Computation of the compressibility factor.
    Z = 1 - (p/T)*(a0 + a1*t + a2*t**2 + (b0 + b1*t)*xv + (c0 + c1*t)*xv**2) + ((p/T)**2)*(d + e*xv**2)
    ## Computation of derivatives of the compressibility factor.
    # Derivative with respect to temperature.
    dZ_dt = (p/T)*((1/T)*(a0 + a1*t + a2*t**2 + (b0 + b1*t)*xv + (c0 + c1*t)*xv**2) - (a1 + 2*a2*t + b1*xv + (b0 + b1*t)*dxv_dt + c1*xv**2 + 2*(c0 + c1*t)*xv*dxv_dt) + 2*(p/T)*(e*xv*dxv_dt - (1/T)*(d + e*xv**2)))
    # Derivative with respect to pressure.
    dZ_dp = (1/T)*(2*(p/T)*(d + e*xv**2 + p*e*xv*dxv_dp) - p*dxv_dp*((b0 + b1*t) + 2*(c0 + c1*t)*xv) - (a0 + a1*t + a2*t**2 + (b0 + b1*t)*xv + (c0 + c1*t)*xv**2))
    # Derivative with respect to relative humidity.
    dZ_dRH = (p/T)*dxv_dRH*(2*(p/T)*e*xv - (b0 + b1*t + 2*(c0 + c1*t)*xv))

    ## Computation of density from moist air state equation taken from formula in reference 2007Picard.
    rho = 3.483740e-3*(p/(Z*T))*(1 - 0.3780*xv)
    ## Computation of derivatives of density.
    # Derivative with respect to temperature.
    drho_dt = -3.48374e-3*(p/(Z*T))*(0.378*dxv_dt + (1 - 0.378*xv)*(Z + T*dZ_dt)/(Z*T))
    # Derivative with respect to pressure.
    drho_dp = 3.48374e-3*(1/(Z*T))*((1-0.378*xv)*(1 - (p/Z)*dZ_dp) - 0.378*p*dxv_dp)
    # Derivative with respect to RH.
    drho_dRH = -3.48374e-3*(p/(Z*T))*(0.378*dxv_dRH + (1 - 0.378*xv)*dZ_dRH/Z)

    ## Computation of viscosity from formula in reference 1985Zuckerwar.
    mu = (84.986 + 7*T + 113.157*xv - T*xv - 3.7501e-03*T**2 - 100.015*xv**2)*1e-08
    ## Computation of derivatives of viscosity.
    # Derivative with respect to temperature.
    dmu_dt = (7 + 113.157*dxv_dt -xv - T*dxv_dt - 2*3.7501e-3*T - 2*100.15*xv*dxv_dt)*1e-08
    # Derivative with respect to pressure.
    dmu_dp = dxv_dp*(113.157 - T - 2*100.15*xv)*1e-08
    # Derivative with respect to RH.
    dmu_dRH = dxv_dRH*(113.157 - T - 2*100.15*xv)*1e-08

    # Return statement.
    if mode=='noder':
        return rho, mu
    else:
        return drho_dt, drho_dp, drho_dRH, dmu_dt, dmu_dp, dmu_dRH

# decay_range() function.
def decay_range(files, vel_comp='u', rtype='energies', x0_M_DM_xiso=(7, 0.015, 1, 20), meas=0, imposeX0=(False, 0)):
    '''Computes data relative to the power-law-like decay range of a turbulence field behind a grid.
    
    The method is intended to compute either the energetic evolution (kinetic energy and turbulent dissipation), the relevant
    length scales (integral, Taylor and Kolmogorov) or the turbulence constants (Saffman and Batchelor). Alternatively, it is
    possible to compute the skewness and kurtosis of the input signals in order to analyse the closeness to isotropic conditions.
    
    -**paramters**, **returns**, **rtypes**
    
    :param files: list of TDMS files containing turbulent data read by the Cobra probe.
    :param vel_comp: the velocity component to which the method is to be applied. Either 'u' (streamwise velocity) or 'U' (overall
    velocity).
    Default is 'u'.
    :param rtype: string parameter determining the data to output. It may be 'iso' (for outputting skewness and kurtosis as functions
    of the downstream non-dimensional positional parameter), 'energies' (for outputting kinetic energy and dissipation evolutions),
    'scales' (for outputting integral length scale, Taylor and Kolmogorov scale evolutions) or 'constants' (for outputting both
    Saffman and Batchelor constants). Alternatively, it may acquire the value 'opt', which outputs the downstream evolution of the
    coefficient for the optimization process of the kinetic energy fit in terms of the virtual origin to be optimized.
    Default is 'energies'.
    :param x0_M_DM_xiso: tuple which indicates, respectively, the initial non-dimensional positional stage, the dimensional mesh
    parameter M (in m), the non-dimensional distance between succesive positional stages, and the non-dimensional positional stage
    for which quasi-isotropic and homogeneous conditions ensue.
    Default is (7, 0.015, 1, 20).
    :param meas: integer value that indicates how many measurements are to be employed in the calculations. It counts the measurements
    from the first one, and trims the last (#measurements - meas) measurements.
    Default is 0, which is to say that all measurements are considered.
    :param imposeX0: tuple(bool, int) that tells whether a given virtual origin is to be set. If imposeX0[0]=True, then the optimization
    process is skipped and the calculations are performed with the provided imposeX0[1] value.
    Default is (False, 0), which is to say that the optimization process is undertaken.
    :returns: if 'rtype' is 'iso' -> non-dimensional parameter, deviation, skewness, kurtosis.
              if 'rtype' is 'energies' -> exponents and coefficients for kinetic energy and dissipation decay fit and virtual origin;
              non-dimensional parameter, kinetic energies, kinetic energies' fit, dissipation.
              if 'rtype' is 'scales' -> exponents and coefficients for length scale fits and virtual origin; non-dimensional parameter,
              integral length scale, square of the Taylor scale, Kolmogorov scale.
              if 'rtype' is 'constants' -> non-dimensional parameter, Saffman constant, Batchelor constant.
              if 'rtype is 'opt' -> non-dimensional parameter, virtual-origin-dependent n coefficient evolutions, optimum n coefficient
              index.
    :rtypes: if 'rtype' is 'iso' -> tuple(array, array, array, array).
             if 'rtype' is 'energies' -> tuple(list(double, double), list(double, double), integer), tuple(array, array, array, array)
             if 'rtype' is 'scales' -> tuple(list(double, double, double), list(double, double, double), integer),
             tuple(array, array, array)
             if 'rtype' is 'constants' -> tuple(array, array, array)
             if 'rtype is 'opt' -> tuple(array, array, integer)
    
    '''
    
    # List that contains the number of measurements in each file of the 'files' variable.
    measurements_list = list()
    # Asserting that 'file' is given in a correct format, or either as a string literal.
    for e, file in enumerate(files):
#         assert type(file) in [TDMSC.TdmsFileRoot, str], "Please provide a valid TDMS file format, or either input it as a string literal."
        # If 'file' is given as a string literal, then change it to the correspondent global variable.
        if type(file) == str:
            files[e] = globals()[file]
        measurements_list.append(sorted([prop for prop in dir(file.groups_original) if "measurement" in prop], key=lambda x: int(x.split("measurement")[-1])))
                                 
    assert measurements_list.count(measurements_list[0]) == len(measurements_list), "Please provide TDMS files with same number of measurements to perform the averaging; otherwise, please process the files individually."
    # Obtaining the set of sorted measurement names from the provided TDMS file (measurement1...measurementN).
    measurements = measurements_list[0]
            
    # Asserting that 'vel_comp' has an admissible value ('u'/'v'/'w'/'U').
    assert vel_comp in ['u', 'q'], "Please provide valid velocity component(s), 'u'/'U'."
        
    # Asserting that 'out' has an admissible value ('iso'/'decay').
    assert rtype in ['iso', 'opt', 'energies', 'scales', 'constants'], "Provide a valid output parameter: 'iso'/'opt'/'energies'/'scales'/'constants'."
        
    # If 'vel_comp' is 'q', then assign it the string 'U'.
    if vel_comp == 'q':
        vel_comp = 'U'                
    
    # Conditional for checking whether the 'meas' parameter is to be changed. If the input parameter's value surpasses the
    # number of measurements in the file, then it is set to that number. Instead, it is kept as it is.
    if meas == 0 or meas > len(measurements) - 1:
        meas = len(measurements) - 1
    else:
        pass
    
    # The 'x0' variable is intended to store the non-dimensional starting point of the test.
    x0 = x0_M_DM_xiso[0]
    # The 'M' variable is intended to store the mesh parameter value.
    M = x0_M_DM_xiso[1]
    # The 'DM' variable is intended to store the mesh spacing parameter value.
    DM = x0_M_DM_xiso[2]
    # The 'xiso' variable is intended to store the spatial stage where quasi-isotropic and homogeneous conditions hold.
    xiso = x0_M_DM_xiso[3]
    # The 'ndx' variable is the array storing the non-dimensional spatial stages tested.
    ndx = np.arange(x0, x0 + len(measurements))
    # The 'xinit' variable stores the value of the initial spatial stage where quasi-isotropic and homogeneous conditions
    # are assumed.
    xinit = (xiso - x0)//DM
    
    Ufixs = np.zeros(len(measurements))
    Us = np.zeros(len(measurements))
    nus = np.zeros(len(measurements))
    qcins = np.zeros(len(measurements))
    for e, file in enumerate(files):
        # The 'Ufixs' variable is intended to store the mean values of the velocities measured by the fix probe.
        Ufixs += np.array(file.groups_added.means_group.means_channels.fix_probe_signal)
        # The 'Us' variable is intended to store the mean values of the velocities measured by the Cobra probe.
        Us += np.array(file.groups_added.means_group.means_channels.cobra_U_signal  )
        # The 'nus' variable is intended to store the kinematic viscosities of the Cobra file.
        nus += np.array([_.mu/_.rho for _ in file.__ref_mags__])
        # The 'qcins' variable is intended to store the kinetic energy signal of the overall velocity vector.
        qcins += np.array(file.groups_added.means_group.means_channels.cobra_uvwcin_signal)
    Ufixs /= len(files)
    Us /= len(files)
    nus /= len(files)
    qcins /= len(files)
    # The 'velcins' variable is intended to store the kinetic energy signals of the velocity component determined by
    # 'vel_comp'.
    velcins = list()
    velcins_ave = np.zeros(len(measurements))                                 
    # The 'ndvelcins' variable is intended to store the non-dimensional kinetic energy signals of the velocity component
    # determined by 'vel_comp'.
    ndvelcins = list()
    ndvelcins_ave = np.zeros(len(measurements))
    # The 'Lambdas' variable is intended to store the integral length scales computed from the integration of the
    # autocorrelation signal.
    Lambdas = list()
    Lambdas_ave = np.zeros(len(measurements))
    # The 'deviation' variable is intended to store the skewness of the velocity fluctuation signal.
    deviation = list()
    deviation_ave = np.zeros(len(measurements))    
    # The 'skewness' variable is intended to store the skewness of the velocity fluctuation signal.
    skewness = list()
    skewness_ave = np.zeros(len(measurements))
    # The 'kurtosis' variable is intended to store the kurtosis of the velocity fluctuation signal.
    kurtosis = list()
    kurtosis_ave = np.zeros(len(measurements))
    
    for f, file in enumerate(files):
            
        velcins = list()
        ndvelcins = list()
        Lambdas = list()
        deviation = list()
        skewness = list()
        kurtosis = list()
        
        # Loop running over the overall set of measurements.
        for e, measurement in enumerate(measurements):
            
            # The 'temp' variable is intended to store the temporal signal of the Cobra probe for the current measurement.
            temp = file.groups_original.__getattribute__(measurement).channels.cobra_signal_time.data
            # The 'dt' variable is intended to store the temporal spacing of the current Cobra measurement.
            dt = temp[1] - temp[0]
            # The 'sampling_freq' variable is intended to store the sampling frequency of the current Cobra measurement.
            sampling_freq = int(1/(dt))

            # The 'vel' variable stores the velocity signal of the current Cobra measurement corresponding to the
            # component determined by 'vel_comp'.
            vel = file.groups_original.__getattribute__(measurement).channels.__getattribute__("cobra_" + vel_comp + "_comp_signal").data
            # The 'velfluc' variable stores the velocity fluctuation signal of the current Cobra measurement
            # corresponding to the component determined by 'vel_comp'.      
            velfluc = file.groups_original.__getattribute__(measurement).channels.__getattribute__("cobra_" + vel_comp + "fluc_comp_signal").data        
            # Computing temporal derivative of fluctuation signal.
            dvelfluc_dt = derivative(velfluc, dt)
            # Computing spatial derivative of fluctuation signal.
            dvelfluc_dx = (1/Us[e])*dvelfluc_dt
            # The 'velcins' variable stores the  turbulent kinetic energy signal of the current Cobra measurement
            # corresponding to the component determined by 'vel_comp'.
            velcins.append((np.average(velfluc**2)))
            # The 'ndvelcins' variable stores the non-dimensional turbulent kinetic energy signal of the current
            # Cobra measurement corresponding to the component determined by 'vel_comp'.
            ndvelcins.append((velcins[e]/Us[e]**2))
            # Adding the deviation value of the velocity fluctuation signal to 'deviation' variable
            deviation.append(np.std(velfluc))
            # Adding the skewness value of the velocity fluctuation signal to 'skewness' variable.
            skewness.append(sc.stats.skew(velfluc, bias=False))
            # Adding the kurtosis of the velocity fluctuation signal to the 'kurtosis' variable.
            kurtosis.append(sc.stats.kurtosis(velfluc, fisher=False))
            # Computing autocorrelation of the fluctuating velocity.
            autocorr = scsg.fftconvolve(velfluc, velfluc[::-1], 'same')
            # Trimming autocorrelation in halves to obtain positive contributions only.
            autocorr = autocorr[len(autocorr)//2:]
            # Scaling autocorrelation, and adding the half-sliced temporal signal to it.
            autocorr = (temp[::2], autocorr / max(autocorr))
            # Finding last index before a negative value of the autocorrelation is found.
            ind = [_ < 0 for _ in autocorr[1]].index(True) - 1
            # Integrating the autocorrelation function portion standing between the initial index and the previously found one.
            Lambdas.append(Ufixs[e]*sc.integrate.simps(autocorr[1][:ind], autocorr[0][:ind]))

        velcins_ave += np.array(velcins)
        ndvelcins_ave += np.array(ndvelcins)
        Lambdas_ave += np.array(Lambdas)
        deviation_ave += np.array(deviation)
        skewness_ave += np.array(skewness)
        kurtosis_ave += np.array(kurtosis)
        
    velcins_ave /= len(files)
    ndvelcins_ave /= len(files)
    Lambdas_ave /= len(files)
    deviation_ave /= len(files)
    skewness_ave /= len(files)
    kurtosis_ave /= len(files)
                                 
    # Trimming the above variables to comply with the input 'meas' parameter.
    Ufixs = Ufixs[:meas]
    Us = Us[:meas]    
    nus = nus[:meas]
    dvelfluc_dt = dvelfluc_dt[:meas]
    dvelfluc_dx = dvelfluc_dx[:meas]
    velcins_ave = velcins_ave[:meas]
    ndvelcins_ave = ndvelcins_ave[:meas]
    deviation_ave = deviation_ave[:meas]
    skewness_ave = skewness_ave[:meas]
    kurtosis_ave = kurtosis_ave[:meas]
    Lambdas_ave = Lambdas_ave[:meas]
    ndx = ndx[:meas]
    
    # If 'rtype' is 'iso', then computing isotropy parameters.
    if rtype == 'iso':
        # Return statement.
        return deviation_ave, skewness_ave, kurtosis_ave, list(ndx)
    # If 'out' is 'decay', then computing the decay law by maximum decay range method (Lavoie 2007).
    else:
        # The 'stds' variable is intended to store the standard devaitions of the potential optimum fits.
        stds = list()
        # The 'Ns' variable is intended to store the evolution of the N coefficient during the optimization process.
        Ns = list()
        # Conditional for checking whether the optimization process is to be undertaken or not.
        if not imposeX0[0]:
            # If 'imposeX0[0]'=False, then the virtual origin is to be found by an optimization process. This process
            # is undertaken for a number of virtual origin values ranging from 0 to x0-1.
            x0List = np.linspace(0, x0-1, x0)
        else:
            # Otherwise, as the optimization process is not to be undertaken, the only number into the list of possible
            # virtual origin candidates is 'imposeX0[1]'.
            x0List = [imposeX0[1]]
        # 'for' loop running over the possible values of the virtual origin x0.            
        for x in x0List:
            # The 'ns' variable is intended to store the evolution of the N coefficient during a particular optimization
            # process; i.e. for a given value of the virtual origin parameter.
            ns = list()
            # 'for' loop running over the measurement 'points' to be fitted, and dropping them one by one from the
            # largest positional values.
            for i in range(len(ndx), 2, -1):
                # Appending the value of the slope of the resultant fit to 'ns'.
                ns.append(least_squares(np.log(ndx[:i] - x), np.log(ndvelcins_ave[:i]))[0])
            # Computing the standard deviation of the homogeneous part of the measurements for the given optimization.
            stds.append(np.std(ns[:len(ns)-xinit]))            
            # Appending the overall evolution of the fit slope coeefficient to 'Ns'.
            Ns.append(ns)
        # Obtaining the index of the minimum standard deviation as computed for the homogeneous part of the measurements.
        ind = stds.index(min(stds))
        # Retrieving the value of the virtual origin based on the minimum standard deviation index.
        x0virt = np.linspace(0, x0-1, x0)[ind] if not imposeX0[0] else imposeX0[1]
        
        # If 'out' is 'opt', then return the optimization curves for each of the virtual origins tested.
        if rtype == 'opt':
            # Conditional for checking whether a virtual origin has been imposed. The returned values are varied accordingly.
            if not imposeX0[0]:
                # Return statement; yielding the non-dimensional positions, the inverted evolutions of the slope coefficients
                # of the fits (so that the indices of the list correspond to the number of measurement points of the fit
                # starting from the nearest positional point), and the optimum curve index.
                return (ndx[:-2], [n[::-1] for n in Ns], ind)
            else:
                print('x0 imposed; optimization process not undertaken.')
                return (ndx[:-2], [n[::-1] for n in Ns], imposeX0[1])
                
        # Obtaining fit coefficients for the non-dimensional kinetic energy.
        n, A = least_squares(np.log(ndx[xinit:] - x0virt), np.log(ndvelcins_ave[xinit:]))[:2]
        # Obtaining fit coefficients for the dimensional kinetic energy.
        nvel, Avel = least_squares(np.log(ndx[xinit:] - x0virt), np.log(velcins_ave[xinit:]))[:2]
        # Fitting dimensional kinetic energy.
        qvelfit = np.e**(Avel)*(ndx[xinit:] - x0virt)**nvel
        # Deriving fit of dimensional kinetic energy with respect to spatial variable.
        dq_dx = derivative(qvelfit, M)        
        
        # Computing the output non-dimensional spatial variable; trimming the array from the onset of quasi-isotropic,
        # homogeneous conditions and substracting the virtual origin.
        ndx_out = ndx[xinit:] - x0virt
        # Computing the output non-dimensional kinetic energy; trimming the array from the onset of quasi-isotropic,
        # homogeneous conditions.
        ndvelcins_out = ndvelcins_ave[xinit:]
        # Computing the output fit of the non-dimensional kinetic energy from the obtained coefficients n, A.
        ndvelcinsfit_out = np.e**A*(ndx[xinit:] - x0virt)**n
        
        # Conditional for checking which dissipation and Taylor scale (mesoscale) computation is ensued; if the
        # requested kinetic energies are those of the streamwise velocity component ('u'), then isotropic conditions
        # are assumed, and the dissipation and Taylor scale are computed as:
        # epsilon = -(3/2*U*(d<u^2>/dX))
        # lamda^2 = 15*nu*<u^2>/epsilon
        if vel_comp == 'u':
            dissipation = -3*(np.array(Us[xinit:])/2)*dq_dx
            lambda_squared = 15*np.array(nus[xinit:])*qvelfit/dissipation
        # If the requested kinetic energies are those of the overall velocity vector ('q'), then:
        # epsilon = -(U/2)*d<q^2>/dX
        # lambda^2 = 5*nu*<q^2>/epsilon
        else:
            dissipation = -(np.array(Us[xinit:])/2)*dq_dx
            lambda_squared = 5*np.array(nus[xinit:])*qvelfit/dissipation                        
        # Non-dimensionalizing dissipation and Taylor scale.
        ndissipation = (M/((np.array(Ufixs[xinit:])**3)))*dissipation
        nlambda_squared = lambda_squared/M**2
        
        # Computing Kolmogorov scale as eta = (nu^3/epsilon)^(0.25).
        eta = (np.array(nus[xinit:])**3/dissipation)**(0.25)
        # Non-dimensionalizing Kolmogorov scale.
        ndeta = eta/M
        # Non-dimensionalizing integral length scale.
        nLambda = np.array(Lambdas_ave[xinit:])/M
        
        # Computing Saffman constant.
        saffman_const = ndvelcins_out*nLambda**3
        # Computing Batchelor constant.
        batchelor_const = ndvelcins_out*nLambda**4
        
        # Conditional for discriminating between outputs.
        if rtype == "energies":
            # If energies (kinetic and dissipation) are to be output, then computing the coefficients of the fit
            # of the dissipation.
            ndis, Adis = least_squares(np.log(ndx_out), np.log(ndissipation))[:2]
            # Return statement.
            return ([n, ndis], [A, Adis], x0virt), (ndx_out, ndvelcins_out, ndvelcinsfit_out, ndissipation)
        elif rtype == "scales":
            # If scales (integral, Taylor and Kolmogorov) are to be output, then computing the coefficients of the
            # fits of those scales.
            nL, AL = least_squares(np.log(ndx_out), np.log(nLambda))[:2]
            nl, Al = least_squares(np.log(ndx_out), np.log(np.sqrt(nlambda_squared)))[:2]
            ne, Ae = least_squares(np.log(ndx_out), np.log(ndeta))[:2]
            # Return statement.
            return ([nL, nl, ne], [AL, Al, Ae], x0virt), (ndx_out, nLambda, nlambda_squared, ndeta)
        elif rtype == "constants":
            # Return statement.
            return (ndx_out, saffman_const, batchelor_const)
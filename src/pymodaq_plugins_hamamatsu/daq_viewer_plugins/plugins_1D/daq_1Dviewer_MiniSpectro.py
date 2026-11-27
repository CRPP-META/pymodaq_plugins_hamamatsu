import numpy as np
from pymodaq.control_modules.viewer_utility_classes import (
    DAQ_Viewer_base,
    comon_parameters,
    main,
)
from pymodaq.utils.data import DataFromPlugins
from pymodaq_data.data import Axis, DataToExport
from pymodaq_gui.parameter import Parameter
from pymodaq_utils.utils import ThreadCommand

from pymodaq_plugins_hamamatsu.hardware.minispectro import Minispectro


class DAQ_1DViewer_MiniSpectro(DAQ_Viewer_base):
    """ Instrument plugin class for Hamamatsu USB Mini-spectrometers.
    
    This object inherits all functionalities to communicate with PyMoDAQ's DAQ_Viewer module through inheritance via
    DAQ_Viewer_base. It makes a bridge between the DAQ_Viewer module and the Python wrapper of a particular instrument.

    No drivers are needed, the plugin should work with Hamamatsu mini-spectrometers connected with USB. On Linux you will
    have to make sure a udev rule is set so you can open the device without root e.g. in /etc/udev/rules.d/99-hamamatsu.rules
    
    The plugin was tested with C10083CA (TM-CCD) mini-spectrometer.

    Tested with PyMoDAQ 5.3 on Windows 11.
    """

    params = comon_parameters + [
        {'title': 'Sensor name', 'name': 'sensor_name', 'type': 'str', 'value': '', 'readonly': True},
        {'title': 'Serial number', 'name': 'serial_number', 'type': 'str', 'value': '', 'readonly': True},
        {'title': 'Pixels', 'name': 'pixel_nb', 'type': 'str', 'value': '', 'readonly': True},
        {'title': 'Integration time (ms)', 'name': 'integration_time', 'type': 'int', 'value': 100, 'min': 5, 'max': 10000,
                        'siPrefix': True, 'tip': 'MIN = 5 ms, MAX = 10000 ms'}
        ]

    def ini_attributes(self):
        self.controller: Minispectro = None
        self.x_axis = None

    def ini_detector(self, controller=None):
        """Detector communication initialization

        Parameters
        ----------
        controller: (object)
            custom object of a PyMoDAQ plugin (Slave case). None if only one actuator/detector by controller
            (Master case)

        Returns
        -------
        info: str
        initialized: bool
            False if initialization failed otherwise True
        """
        if self.is_master:
            self.controller = Minispectro()
            initialized = True
        else:
            self.controller = controller
            # initialized = True

        self.settings.child('sensor_name').setValue(self.controller.sensor_name)
        self.settings.child('serial_number').setValue(self.controller.serial)
        self.settings.child('pixel_nb').setValue(len(self.controller.wl))

        data_x_axis = np.array(self.controller.wl)*1e-9  # Axis' axes needs array + conversion to nm
        self.x_axis = Axis(data=data_x_axis, label='Wavelength', units='m', index=0)

        # Initialize viewers panel with the future type of data
        self.dte_signal_temp.emit(DataToExport(name='MiniSpectro',
                                               data=[DataFromPlugins(name='Mini-spectrometer',
                                                                     data=[np.array([0 for _ in range(len(self.controller.wl))])],
                                                                     dim='Data1D', labels=['Spectrometer'],
                                                                     axes=[self.x_axis])]))

        info = "Minispectrometer initialized"
        initialized = True
        return info, initialized

    def close(self):
        """Terminate the communication protocol"""
        if self.is_master and self.controller is not None:
            self.controller.close()

    def grab_data(self, Naverage=1, **kwargs):
        """Start a grab from the detector

        Parameters
        ----------
        Naverage: int
            Number of hardware averaging (if hardware averaging is possible, self.hardware_averaging should be set to
            True in class preamble and you should code this implementation)
        kwargs: dict
            others optionals arguments
        """
        # Synchrone version (blocking function)
        data_tot = self.controller.get_spectrum()
        self.dte_signal.emit(DataToExport(name='minispectro',
                                          data=[DataFromPlugins(name='Mini-spectrometer',
                                                                data=data_tot,
                                                                dim='Data1D',
                                                                labels=['Spectrometer'],
                                                                axes=[self.x_axis])]))

    def stop(self):
        """
        Stop the current grab by emitting a status.
        """
        self.emit_status(ThreadCommand('Update_Status', ['Useless for mini-spectrometers (no existing stop function)']))

    def commit_settings(self, param: Parameter):
        """Apply the consequences of a change of value in the detector settings

        Parameters
        ----------
        param: Parameter
            A given parameter (within detector_settings) whose value has been changed by the user
        """
        if param.name() == "integration_time":
            self.controller.set_integration_time(us=int(self.settings['integration_time']*1e3))  # Method needs µs

if __name__ == '__main__':
    main(__file__)
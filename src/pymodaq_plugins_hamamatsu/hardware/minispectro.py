"""Hamamatsu TM-series mini-spectrometer (C10082 / C10083) driver, Python 3 + pyusb.

Port of HamamatsuMiniSpectrometer.py (usbspecdrivers on SourceForge) by
Carsten Frank and Wolfgang Schoenfeld, GPL v2. This port keeps the same
license (GPL v2) and is based on their reverse-engineered USB protocol.

Needs: pip install pyusb numpy (and libusb-1.0 on the system)
On Linux → needs a udev rule so you can open the device without root,
e.g. in /etc/udev/rules.d/99-hamamatsu.rules:
SUBSYSTEM=="usb", ATTR{idVendor}=="0661", MODE="0666"
"""

import argparse
import time

import numpy as np
import usb.core
import usb.util

VENDOR_ID = 0x0661
PRODUCT_IDS = {
    0x2906: "C9407MA",
    0x2909: "C10082/C10083",
}

FRAME_BYTES = 0x1040            # one frame = 0x820 big-endian 16-bit words
FRAME_WORDS = FRAME_BYTES // 2

# vendor control requests
REQ_READ_ROM = 0x04             # IN : wValue = ROM address
REQ_SET_INTEGRATION = 0x0A      # OUT: 4 bytes, scrambled order (see below)
REQ_GET_INTEGRATION = 0x0B      # IN : 4 bytes, big-endian

ROM_SERIAL = 0x1D4              # 8 bytes ASCII
ROM_SENSOR = 0x1DC              # 16 bytes ASCII
ROM_COEFF = [0x000, 0x010, 0x020, 0x030, 0x040, 0x050]  # a0..a5, 16 bytes ASCII floats

MIN_INT_US = 10_000
MAX_INT_US = 10_000_000


class Minispectro:
    def __init__(self, serial=None, product_id=None, integration_us=MIN_INT_US):
        self.dev = self._find(serial, product_id)
        self._claim()

        self.serial = self._rom_text(ROM_SERIAL, 8)
        self.sensor_name = self._rom_text(ROM_SENSOR, 16)
        self.coeffs = [float(self._rom_text(a, 16)) for a in ROM_COEFF]

        # S10420-1006/-1106 CCDs: first spectral pixel is 10 pixels after the
        # start index. Other sensors (S8377/S8378): no offset.
        self.pixel_offset = 10 if self.sensor_name in ("S10420-1106", "S10420-1006") else 0

        self.ep = self._find_in_endpoint()
        self.wl = None
        self.integration_us = 0
        if not self.set_integration_time(integration_us):
            raise RuntimeError("could not set integration time")
        self.get_spectrum()     # first frame also fills self.wl

    # ---- discovery / setup -------------------------------------------------
    @staticmethod
    def _find(serial, product_id):
        pids = [product_id] if product_id else list(PRODUCT_IDS)
        devs = [d for d in usb.core.find(find_all=True, idVendor=VENDOR_ID)
                if d.idProduct in pids]
        if not devs:
            raise RuntimeError(
                "No Hamamatsu spectrometer found (VID 0x0661). Check `lsusb`; "
                "if the PID is not in PRODUCT_IDS, pass product_id=0x....")
        if serial is None:
            return devs[0]
        for d in devs:
            d.set_configuration()
            if Minispectro._read_text(d, ROM_SERIAL, 8) == serial:
                return d
        raise RuntimeError(f"No spectrometer with serial {serial!r}")

    def _claim(self):
        try:
            if self.dev.is_kernel_driver_active(0):
                self.dev.detach_kernel_driver(0)
        except (NotImplementedError, usb.core.USBError):
            pass
        self.dev.set_configuration()

    def _find_in_endpoint(self):
        intf = self.dev.get_active_configuration()[(0, 0)]
        eps = [e for e in intf
               if usb.util.endpoint_direction(e.bEndpointAddress) == usb.util.ENDPOINT_IN]
        bulk = [e for e in eps
                if usb.util.endpoint_type(e.bmAttributes) == usb.util.ENDPOINT_TYPE_BULK]
        if not (bulk or eps):
            raise RuntimeError("no IN endpoint found")
        return (bulk or eps)[0]

    # ---- ROM / control transfers ------------------------------------------
    @staticmethod
    def _read_text(dev, addr, length):
        raw = dev.ctrl_transfer(0xC0, REQ_READ_ROM, addr, 0, length, timeout=2000)
        return bytes(raw).decode("ascii", "ignore").strip("\x00 \r\n\t")

    def _rom_text(self, addr, length):
        return self._read_text(self.dev, addr, length)

    def set_integration_time(self, us, verify=True):
        """Integration time in microseconds, 10 ms .. 10 s."""
        us = int(us)
        if not MIN_INT_US <= us <= MAX_INT_US:
            raise ValueError(f"integration time must be {MIN_INT_US}..{MAX_INT_US} us")
        payload = bytes([(us >> 16) & 0xFF, (us >> 24) & 0xFF, us & 0xFF, (us >> 8) & 0xFF])
        t0 = time.monotonic()
        while True:
            self.dev.ctrl_transfer(0x40, REQ_SET_INTEGRATION, 1, 0, payload, timeout=1000)
            if not verify:
                break
            time.sleep(0.01)
            if self.get_integration_time() == us:
                break
            if time.monotonic() - t0 > 1.0:
                return False
        self.integration_us = us
        return True

    def get_integration_time(self):
        for _ in range(50):
            res = bytes(self.dev.ctrl_transfer(0xC0, REQ_GET_INTEGRATION, 1, 0, 4, timeout=2000))
            if len(res) == 4:
                return int.from_bytes(res, "big")
            time.sleep(0.01)
        raise RuntimeError("bad answer to get_integration_time")

    # ---- frame acquisition -------------------------------------------------
    def _read_words(self, nbytes, timeout_ms):
        buf = bytearray()
        deadline = time.monotonic() + timeout_ms / 1000.0
        while len(buf) < nbytes:
            if time.monotonic() > deadline:
                raise TimeoutError("USB read timed out")
            buf += bytes(self.ep.read(nbytes - len(buf), timeout=timeout_ms))
        return np.frombuffer(bytes(buf[:nbytes]), dtype=">u2").astype(np.int64)

    @staticmethod
    def _valid_header(w, off=0):
        if off + 6 > len(w):
            return False
        flag, idx, _active, start, end, npix = (int(x) for x in w[off:off + 6])
        return flag == 2 and 0 <= idx <= 2 and (npix == end - start or npix == end)

    def get_spectrum(self, timeout=None):
        """Return one spectrum as an int array (raw counts, one value per pixel).

        Frame header words: [0]=2 (valid; 3 = no external power), [1]=0..2,
        [2]=active pixels, [3]=start index, [4]=end index, [5]=pixels, [8]=counter.
        """
        if timeout is None:
            timeout = self.integration_us / 1e6 * 8.0 + 1.0
        t0 = time.monotonic()
        read_timeout_ms = int(timeout * 1000) + 1000

        while True:
            if time.monotonic() - t0 > timeout:
                raise TimeoutError("no valid spectrum received")
            w = self._read_words(FRAME_BYTES, read_timeout_ms)

            if not w.any():                       # device not ready yet
                time.sleep(0.002)
                continue
            if w[0] == 3:
                raise RuntimeError("Spectrometer has no power: USB is connected, "
                                   "use the AC adapter too.")
            if self._valid_header(w):
                break

            # stream out of sync: find the next header and drop the bytes before it
            offset = next((o for o in range(len(w)) if w[o] == 2 and self._valid_header(w, o)),
                          len(w))
            self._read_words(offset * 2, read_timeout_ms)   # discard, then read a fresh frame

        active, start = int(w[2]), int(w[3]) + self.pixel_offset
        if self.wl is None:
            pix = np.arange(1, active + 1, dtype=float)
            self.wl = sum(c * pix ** k for k, c in enumerate(self.coeffs))
        return w[start:start + active]

    def read_spectrum(self, averages=1):
        """Return (wavelengths_nm, counts); counts averaged over `averages` frames."""
        acc = np.zeros_like(self.get_spectrum(), dtype=float)
        for _ in range(averages):
            acc += self.get_spectrum()
        return self.wl, acc / averages

    def close(self):
        usb.util.dispose_resources(self.dev)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser(description="Acquire one spectrum from a Hamamatsu mini-spectrometer")
    ap.add_argument("-t", "--integration-ms", type=float, default=10.0)
    ap.add_argument("-n", "--averages", type=int, default=1)
    ap.add_argument("-o", "--output", help="write wavelength,counts CSV")
    ap.add_argument("--pid", type=lambda s: int(s, 0), help="USB product id, e.g. 0x2909")
    args = ap.parse_args()

    spec = Minispectro(product_id=args.pid)
    print(f"serial {spec.serial}, sensor {spec.sensor_name}, a0={spec.coeffs[0]:.3f}")
    spec.set_integration_time(args.integration_ms * 1000)
    wl, counts = spec.read_spectrum(args.averages)

    # for i in range(100):
    #     wl, counts = spec.read_spectrum(args.averages)
    #     print(f"max_count = {int(counts.max())} at {wl[counts.argmax()]:.2f} nm")
        # print(f"{len(wl)} pixels, {wl[0]:.2f}-{wl[-1]:.2f} nm, peak {int(counts.max())} counts at {wl[counts.argmax()]:.2f} nm")
    # if args.output:
    #     np.savetxt(args.output, np.c_[wl, counts], delimiter=",", header="wavelength_nm,counts")
    # spec.close()

    # plt.plot(wl, counts)
    # plt.show()
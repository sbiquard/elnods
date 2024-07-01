import numpy as np
from typing import Optional


THINFP = 640
FOCALPLANE = "focalplanes/focalplane_SAT1_SAT_f095.h5"
SCHEDULE = "schedules/pole_schedule_sat.1ces.txt"


def simulate_data(
    moves: list[float],
    comm,
    noise: bool = False,
    elevation_noise: bool = False,
    atm_fluctuations: bool = True,
    scramble_gains: bool = False,
    thinfp: Optional[int] = None,
    realization: int = 0,
):
    import toast
    import toast.io
    import toast.schedule
    import toast.ops

    from astropy import units as u

    # Load a generic focalplane file.
    if thinfp is None:
        thinfp = THINFP
    focalplane = toast.instrument.Focalplane(thinfp=thinfp)
    with toast.io.H5File(FOCALPLANE, "r", comm=comm, force_serial=True) as f:
        focalplane.load_hdf5(f.handle, comm=comm)

    # Load the schedule file
    schedule = toast.schedule.GroundSchedule()
    schedule.read(SCHEDULE, comm=comm)

    # Create a telescope for the simulation.
    site = toast.instrument.GroundSite(
        schedule.site_name,
        schedule.site_lat,
        schedule.site_lon,
        schedule.site_alt,
    )
    telescope = toast.instrument.Telescope(
        schedule.telescope_name, focalplane=focalplane, site=site
    )

    # Create the toast communicator.  Use the default of one group.
    toast_comm = toast.Comm(world=comm)

    # Create the (initially empty) data
    data = toast.Data(comm=toast_comm)

    ### Simulate data

    weather = schedule.site_name.lower()
    moves = [x * u.degree for x in moves]

    # Simulate the telescope pointing
    sim_ground = toast.ops.SimGround(
        telescope=telescope,
        schedule=schedule,
        detset_key="pixel",
        weather=weather,
        elnod_start=True,
        elnods=moves,
        realization=realization,
    )

    # Set up detector pointing.  This just uses the focalplane offsets.
    det_pointing_azel = toast.ops.PointingDetectorSimple(
        boresight=sim_ground.boresight_azel, quats="quats_azel"
    )
    # det_pointing_radec = toast.ops.PointingDetectorSimple(
    #     boresight=sim_ground.boresight_radec, quats="quats_radec"
    # )

    # Construct a "perfect" noise model just from the focalplane parameters
    default_model = toast.ops.DefaultNoiseModel()

    # Elevation-modulated noise model.
    elevation_model = toast.ops.ElevationNoise(
        noise_model=default_model.noise_model,
        detector_pointing=det_pointing_azel,
        view=det_pointing_azel.view,
    )

    # Simulate detector noise and accumulate.
    sim_noise = toast.ops.SimNoise(realization=realization)

    # Simulate atmosphere signal
    sim_atm = toast.ops.SimAtmosphere(
        detector_pointing=det_pointing_azel,
        cache_dir="atm_cache",
        realization=realization,
    )
    if not atm_fluctuations:
        # disable atmosphere fluctuations
        sim_atm.gain = 0

    # Scramble the detector gains
    scrambler = toast.ops.GainScrambler(
        sigma=1e-1, store=True, realization=realization
    )

    # Build the final list of operators
    ops = [sim_ground, det_pointing_azel]
    if noise:
        ops.append(default_model)
        if elevation_noise:
            ops.append(elevation_model)
        ops.append(sim_noise)
    ops.append(sim_atm)
    if scramble_gains:
        ops.append(scrambler)

    # Build a pipeline from those operators and run it
    print(f"Operators: {[o.name for o in ops]}")
    pipe = toast.ops.Pipeline(operators=ops)
    pipe.apply(data)

    return data


def get_azel_from_quat(azel_quat):
    import toast.qarray as qa

    # Convert Az/El quaternion of the detector back into
    # angles from the simulation.
    theta, phi, _ = qa.to_iso_angles(azel_quat)

    # Azimuth is measured in the opposite direction
    # than longitude
    az = np.asarray(2 * np.pi - phi)
    el = np.asarray(np.pi / 2 - theta)

    return az, el


def get_el_from_quat(azel_quat):
    import toast.qarray as qa
    theta, _, _ = qa.to_iso_angles(azel_quat)
    return np.asarray(np.pi / 2 - theta)


def pairwise(iterable):
    """s -> (s0,s1), (s2,s3), (s4, s5), ..."""
    a = iter(iterable)
    return zip(a, a)


def div_mean(a):
    return a / np.mean(a, keepdims=True)


def rdiff(a, b):
    return (a - b) / b

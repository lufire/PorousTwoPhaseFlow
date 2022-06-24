import math

boundary_conditions = \
    {
        'current_density': 10000.0,
        'operating_voltage': 0.8,
        'channel_temperature': 343.15,
        'channel_humidity': 1.0,
        'oxygen_fraction': 0.75,
        'gdl_channel_saturation': 0.001,
        'channel_pressure': 101325.0,
        'cl_gdl_liquid_water_fraction': 0.0,
        'cathode_heat_flux_fraction': 0.7
    }

domain = \
    {
        'width': 2e-3,
        'nx': 100,
        'ny': 10,
    }

fluid_dict = \
    {
        "name": "Cathode Gas Mixture",
        "components": {
            "O2": {
                "state": "gas",
                "molar_fraction": 0.21
            },
            "N2": {
                "state": "gas",
                "molar_fraction": 0.79
            },
            "H2O": {
                "state": "gas-liquid",
                "molar_fraction": 0.0
            }
        },
        "humidity": 0.5,
        "temperature": 343.15,
        "pressure": 101325.0,
    }

porous_dict = \
    {
        "name": "Cathode GDL",
        "type": "CarbonPaper",
        "thickness": 200e-6,
        "porosity": 0.78,
        "bruggemann_coefficient": 1.5,
        "permeability": (1.8e-11, 1.8e-11, 1.8e-11),
        "thermal_conductivity": (28.4, 2.8),
        "pore_radius": 1e-5,
        "saturation_model":
            {  # "leverett" or "psd"
                "type": "leverett",
                "leverett":
                    {
                        "type": "leverett",
                        "contact_angle": 120.0
                    },
                "psd":
                    {
                        "type": "psd",
                        "r": [[14.20e-6, 34.00e-6], [14.20e-6, 34.00e-6]],
                        "F": [0.08, 0.92],  # [F_HI, F_HO]
                        "f": [[0.28, 0.72], [0.28, 0.72]],
                        "s": [[0.35, 1.0], [0.35, 1.0]],
                        "contact_angle": [80.0, 120.0]  # [theta_HI, theta_HO]
                    }

            }
    }

evaporation_dict = \
    {
        "name": "Evaporation Model",
        "type": "HertzKnudsenSchrage",
        "evaporation_coefficient": 0.37,
    }

electrode_dict = \
    {
        "name": "Cathode Electrode",
        "reaction_stoichiometry":
            [
                -1.0,
                0.0,
                2.0
            ],
        "charge_number": 4.0,
        "thermoneutral_voltage": 1.482

    }

numerical_dict = \
    {
        "minimum_iterations": 10,
        "maximum_iterations": 500,
        "error_tolerance": 1e-7,
        "under_relaxation_factor": [[300, 1000], [0.001, 0.001]]
    }

SQRT_2 = math.sqrt(2.0)
SQRT_2PI = math.sqrt(2.0 * math.pi)
SMALL = 1e-10

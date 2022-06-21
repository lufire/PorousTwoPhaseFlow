import math

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
        "nodes": (10, 5)
    }

porous_dict = \
    {
        "name": "Cathode GDL",
        "thickness": 200e-6,
        "porosity": 0.78,
        "tortuosity_factor": 1.5,
        "permeability": (1.8e-11, 1.8e-11, 1.8e-11),
        "thermal_conductivity": (28.4, 2.8),
        "pore_radius": 1e-5,
        "nodes": (10, 5)
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

saturation_model_dict = \
    {  # "leverett" or "psd"
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

SQRT_2 = math.sqrt(2.0)
SQRT_2PI = math.sqrt(2.0 * math.pi)
SMALL = 1e-10

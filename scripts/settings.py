boundary_conditions = \
    {
        'avg_current_density': 20000.0,
        'operating_voltage': 0.5,
        'channel_temperature': 343.15,
        'channel_humidity': 1.0,
        'oxygen_fraction': 1.0,
        'gdl_channel_saturation': 0.001,
        'channel_pressure': 101325.0,
        'cl_gdl_liquid_water_fraction': 1.0,
        'cathode_heat_flux_fraction': 0.7
    }

domain = \
    {
        'width': 1e-3,
        'nx': 50,
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
        "humidity": 1.0,
        "temperature": 313.15,
        "pressure": 101325.0,
    }

porous_dict = \
    {
        "name": "Cathode GDL",
        "type": "CarbonPaper",
        "thickness": 200e-6,
        "porosity": 0.8,
        "bruggemann_coefficient": 1.5,
        "permeability": (1.0e-12, 1.0e-12, 1.0e-12),
        "thermal_conductivity": (28.4, 2.8),
        "pore_radius": 1e-5,
        "relative_permeability_exponent": 3.0,
        "saturation_model":
            {  # "leverett", "psd", "imbibition_drainage",
                # "gostick_correlation", "data_table", or "psd"
                "type": "leverett",
                "leverett":
                    {
                        "type": "leverett",
                        "contact_angle": 120.0,
                        # "precalculate_pressure": True,
                        # "precalculate_gradient": True
                    },
                "data_table":
                    {
                        "type": "data_table",
                        "data_format": "file",
                        "file_path": r"C:\Users\lukas\Desktop\test.csv"
                    },
                "psd":
                    {
                        "type": "psd",
                        "r": [[14.20e-6, 34.00e-6], [14.20e-6, 34.00e-6]],
                        "F": [0.08, 0.92],  # [F_HI, F_HO]
                        "f": [[0.28, 0.72], [0.28, 0.72]],
                        "s": [[0.35, 1.0], [0.35, 1.0]],
                        "contact_angle": [80.0, 120.0]
                    },
                "imbibition_drainage":
                    {  # "leverett" or "psd"
                        "type": "imbibition_drainage",
                        "imbibition_model":
                            {
                                "type": "gostick_correlation",
                                "maximum_water_saturation": 0.99,
                                "residual_water_saturation": 0.1,
                                "f": [0.25, 0.75],
                                "m": [250, 200],
                                "n": [0.4, 0.5],
                                "P_C_b": [101500.0, 107500.0],
                                "precalculate_pressure": True,
                                "precalculate_gradient": True
                            },
                        "drainage_model":
                            {
                                "type": "gostick_correlation",
                                "maximum_water_saturation": 0.99,
                                "residual_water_saturation": 0.1,
                                "f": [1.0],
                                "m": [150],
                                "n": [1.0],
                                "P_C_b": [105000.0],
                                "precalculate_pressure": True,
                                "precalculate_gradient": True
                            }
                    },
                "gostick_correlation":
                    {
                        "type": "gostick_correlation",
                        "maximum_saturation": 1.0,
                        "residual_saturation": 0.0,
                        "f": [0.25, 0.75],
                        "m": [250, 200],
                        "n": [0.3, 0.5],
                        "P_C_b": [101500.0, 108000.0]
                        # "type": "gostick_correlation",
                        # "maximum_water_saturation": 0.99,
                        # "residual_water_saturation": 0.1,
                        # "f": [1.0],
                        # "m": [150],
                        # "n": [1.0],
                        # "P_C_b": [105000.0],
                        # "precalculate_pressure": True,
                        # "precalculate_gradient": True
                    },
            }
    }

# evaporation_dict = \
#     {
#         "name": "Evaporation Model",
#         "type": "HertzKnudsenSchrage",
#         "evaporation_coefficient": 0.37,
#     }
evaporation_dict = \
    {
        "name": "Evaporation Model",
        "type": "WangSi",
        "evaporation_coefficient": 1e-4,
        "condensation_coefficient": 5000
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
        "under_relaxation_factor":
            {
                "saturation": [[500, 1000], [0.3, 0.1]],
                "temperature": 0.95,
                "concentration": 0.95,
                "pressure": 0.95
             }
    }


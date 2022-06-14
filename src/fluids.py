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


def calc_surface_tension(temperature):
    return 0.07275 * (1.0 - 0.002 * (temperature - 291.0))

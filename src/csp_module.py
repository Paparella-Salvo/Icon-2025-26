from constraint import Problem

class WeatherCSP:
    def __init__(self):
        # Inizializza un nuovo problema ogni volta che crei l'oggetto
        self.problem = Problem()

    def solve(self, weather, uv, temp):
        # Ricrea il problema da zero (reset corretto)
        self.problem = Problem()

        # Variabili del CSP
        self.problem.addVariable("Activity", ["Hiking", "Beach", "IndoorGym", "CityWalk"])
        self.problem.addVariable("Outfit", ["LightClothes", "Jacket", "Waterproof", "UVProtection"])

        # --- Vincoli basati sul meteo ---
        if weather == "Rainy":
            self.problem.addConstraint(lambda a: a != "Hiking", ["Activity"])
            self.problem.addConstraint(lambda o: o == "Waterproof", ["Outfit"])

        if weather == "Snowy":
            self.problem.addConstraint(lambda o: o == "Jacket", ["Outfit"])
            self.problem.addConstraint(lambda a: a != "Beach", ["Activity"])

        if weather == "Sunny":
            self.problem.addConstraint(lambda a: a != "IndoorGym", ["Activity"])

        # --- Vincoli basati sull'UV ---
        if uv == "high":
            self.problem.addConstraint(lambda o: o == "UVProtection", ["Outfit"])

        # --- Vincoli basati sulla temperatura ---
        if temp == "very_cold":
            self.problem.addConstraint(lambda o: o == "Jacket", ["Outfit"])

        # Risoluzione
        solutions = self.problem.getSolutions()
        return solutions[0] if solutions else None

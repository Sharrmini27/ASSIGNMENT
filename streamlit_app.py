import streamlit as st
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import csv

# ---------- READ CSV ----------
def read_csv_to_dict(file_path):
    program_ratings = {}
    try:
        with open(file_path, mode='r', newline='') as file:
            reader = csv.reader(file)
            header = next(reader)
            for row in reader:
                program = row[0]
                ratings = [float(x) for x in row[1:] if x != ""]
                program_ratings[program] = ratings
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found.")
        st.info("Using fallback sample data.")
        program_ratings = {
            'News': [0.8, 0.6, 0.7, 0.5, 0.4],
            'Movie': [0.4, 0.5, 0.6, 0.7, 0.8],
            'Sports': [0.7, 0.8, 0.5, 0.6, 0.4]
        }
    return program_ratings

# ---------- GA FUNCTIONS ----------
def fitness_function(schedule, ratings_data, schedule_length):
    total_rating = 0
    for time_slot, program in enumerate(schedule):
        if program in ratings_data:
            if time_slot < len(ratings_data[program]):
                total_rating += ratings_data[program][time_slot]
    return total_rating

def create_random_schedule(all_programs, schedule_length):
    return [random.choice(all_programs) for _ in range(schedule_length)]

def crossover(schedule1, schedule2, schedule_length):
    if len(schedule1) < 2 or len(schedule2) < 2:
        return schedule1, schedule2
    point = random.randint(1, schedule_length - 1)
    child1 = schedule1[:point] + schedule2[point:]
    child2 = schedule2[:point] + schedule1[point:]
    return child1, child2

def mutate(schedule, all_programs, schedule_length):
    schedule_copy = schedule.copy()
    mutation_point = random.randint(0, schedule_length - 1)
    new_program = random.choice(all_programs)
    schedule_copy[mutation_point] = new_program
    return schedule_copy

def genetic_algorithm(ratings_data, all_programs, schedule_length,
                      generations=100, population_size=50,
                      crossover_rate=0.8, mutation_rate=0.2, elitism_size=2):
    population = [create_random_schedule(all_programs, schedule_length) for _ in range(population_size)]
    best_schedule_ever = []
    best_fitness_ever = 0

    for generation in range(generations):
        pop_with_fitness = []
        for schedule in population:
            fitness = fitness_function(schedule, ratings_data, schedule_length)
            pop_with_fitness.append((schedule, fitness))

            if fitness > best_fitness_ever:
                best_fitness_ever = fitness
                best_schedule_ever = schedule

        pop_with_fitness.sort(key=lambda x: x[1], reverse=True)
        new_population = [x[0] for x in pop_with_fitness[:elitism_size]]

        while len(new_population) < population_size:
            parent1 = random.choice(pop_with_fitness[:population_size // 2])[0]
            parent2 = random.choice(pop_with_fitness[:population_size // 2])[0]

            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2, schedule_length)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            if random.random() < mutation_rate:
                child1 = mutate(child1, all_programs, schedule_length)
            if random.random() < mutation_rate:
                child2 = mutate(child2, all_programs, schedule_length)

            new_population.append(child1)
            if len(new_population) < population_size:
                new_population.append(child2)

        population = new_population

    return best_schedule_ever, best_fitness_ever


# ---------- STREAMLIT UI ----------
st.title("📺 TV Program Scheduling Optimizer (Genetic Algorithm)")

file_path = 'program_ratings (2).csv'
ratings = read_csv_to_dict(file_path)

try:
    df_display = pd.read_csv(file_path)
    st.subheader("📊 Program Ratings Dataset")
    st.dataframe(df_display)
except FileNotFoundError:
    st.error(f"Could not find {file_path} to display.")

if ratings:
    all_programs = list(ratings.keys())
    all_time_slots = list(range(6, 24))
    SCHEDULE_LENGTH = len(all_time_slots)

    st.write(f"Loaded {len(all_programs)} programs.")
    st.write(f"Optimizing for {SCHEDULE_LENGTH} hourly slots (6 AM – 11 PM).")

    # Sidebar input
    st.sidebar.header("⚙️ GA Parameters (3 Trials)")

    co_r_1, mut_r_1 = 0.8, 0.1
    co_r_2, mut_r_2 = 0.7, 0.2
    co_r_3, mut_r_3 = 0.9, 0.3

    if st.sidebar.button("🚀 Run All 3 Trials"):
        for i, (cr, mr, seed) in enumerate([(co_r_1, mut_r_1, 10), (co_r_2, mut_r_2, 20), (co_r_3, mut_r_3, 30)], start=1):
            random.seed(seed)
            np.random.seed(seed)
            st.subheader(f"Trial {i} Results")
            st.write(f"Crossover Rate = {cr}, Mutation Rate = {mr}")
            schedule, fitness = genetic_algorithm(ratings, all_programs, SCHEDULE_LENGTH,
                                                  crossover_rate=cr, mutation_rate=mr)
            df_result = pd.DataFrame({
                "Time Slot": [f"{h:02d}:00" for h in all_time_slots],
                "Program": schedule
            })
            st.dataframe(df_result)
            st.success(f"Best Fitness Score: {fitness:.2f}")
            st.markdown("---")
else:
    st.error("Could not load any program data.")


''' HANDOUT 2: EVOLUTIONARY ALGORITHM '''
import numpy as np
import copy
import sys
import random
import matplotlib.pyplot as plt
import textwrap


''' DATA INITIALIZATION AND OTHER FUNCTIONS''' 

def assign_labels(names,hours): #Generate the specified number of hours for each name in 'names'
    labels = []
    for name, hour in zip(names, hours):
        for i in range(hour):
            nameLabel = f'{name}'
            labels.append(nameLabel)

    return labels

def assign_labels2(names, hours):
    labels = []
    sorted_hours_names = sorted(zip(hours, names))

    min_hour = sorted_hours_names[0][0]
    # First, assign an equal number of hours to all names
    for hour, name in sorted_hours_names:
        labels.extend([name]*min_hour)
    # Then, assign the remaining hours from the name with the most hours
    for hour, name in sorted_hours_names:
        remaining_hours = hour - min_hour
        if remaining_hours > 0:
            labels.extend([name]*remaining_hours)

    return labels

def total_subject_hours(hoursClass): #Calculate the total number of hours for each subject across all courses
    num_rows = len(hoursClass)
    num_columns = len(hoursClass[0]) if num_rows > 0 else 0
    
    total_sum = [0] * num_columns
    
    for row in hoursClass:
        for index_column, value in enumerate(row):
            total_sum[index_column] += value
    return total_sum    

def assign_localization(subjects, teachers): #Create an index to track the total number of hours to be assigned in the schedules
    index = ['Empty']
    for subject, teacher in zip(subjects,teachers):
        name = f'{subject} with {teacher}'
        index.append(name)
        
    return index

def assign_dictionaries(Index): #Generate dictionaries to map the meaning of integers within the schedules
    strings = {}
    counts = 0
    
    for course in Index:
        if course not in strings:
            strings[course] = counts
            counts += 1
    numbers = {v: k for k, v in strings.items()}
    
    return strings, numbers #Provide results to access the dictionaries, both with the numbers and the corresponding strings

def calculate_max_hours(hoursClass): #Calculate the maximum number of hours that need to be taught in a single course
    sum_rows = [sum(row) for row in hoursClass]
    max_number = max(sum_rows)
    
    return max_number

def generate_subjects_dict(dict_by_numbers): #Locate and combine subjects that are repeated and taught by two different teachers
    # Generate the subjects dictionary
    subjects = {}
    for number, subject in dict_by_numbers.items():
        if number == 0:  # Skip '0'
            continue
        subject = subject.split(' with ')[0]  # Get the name of the subject
        if subject not in subjects:
            subjects[subject] = []
        subjects[subject].append(number)
    return subjects

def choose_best_prospect(teachers, dict_by_numbers, population):
    fitness = population_fitness_function(teachers, dict_by_numbers, population)
    minimum = min(fitness)
    min_index = fitness.index(minimum)
    
    best_schedule = population[min_index]
    
    return best_schedule

''' GENERATION OF THE INITIAL POPULATION '''
def create_class_schedule(hoursClass, hours_per_week, totalHours, Index, dict_by_strings):
    schedules = []  
    totalHour = copy.deepcopy(totalHours)
    temp = copy.deepcopy(Index)
    
    for classes in hoursClass:

        index_init = 1
        schedule = np.zeros((hours_per_week, 5))
        for number_of_subjects, count in zip(classes,range(len(totalHour))): #number of subjects = 2
            for i in range(number_of_subjects):
                random_row = np.random.randint(0, hours_per_week)
                random_column = np.random.randint(0, 5)
                while schedule[random_row,random_column] != 0.0:
                    random_row = np.random.randint(0, hours_per_week)
                    random_column = np.random.randint(0, 5)
                element = temp.pop(index_init)
                number = dict_by_strings[element]
                schedule[random_row, random_column] = number
                totalHour[count]-=1
            index_init += totalHour[count]
                
        schedules.append(schedule)  
        
    return schedules

def generate_schedules_aleatorially(dict_by_strings, hoursClass, hours_per_week, population_size = 1):
    schedules = {}
    for i in range(population_size):
        individual_schedules = []
        for _ in range(len(hoursClass)):
            schedule = np.random.choice(list(dict_by_strings.values()), (hours_per_week, 5))
            individual_schedules.append(schedule)
        schedules[i] = individual_schedules
    return schedules if population_size != 1 else individual_schedules


def generate_population(population_size, hoursClass, hours_per_week, totalHours, Index, dict_by_strings):
    population = {}

    for i in range(population_size):
        individual = create_class_schedule(hoursClass, hours_per_week, totalHours, Index, dict_by_strings)
        
        population[i] = individual
    return population

def generate_mix_population(population_size, hoursClass, hours_per_week, totalHours, Index, dict_by_strings):
    population = {}

    for i in range(population_size):
        # If i is even, call the function create_class_schedule.
        if i % 2 == 0:
            individual = create_class_schedule(hoursClass, hours_per_week, totalHours, Index, dict_by_strings)
        #If i is odd, call the function generate_schedules_aleatorially
        else:
            individual = generate_schedules_aleatorially(dict_by_strings, hoursClass, hours_per_week)
        population[i] = individual
    return population



''' CONSTRAINTS '''
    #HARD CONSTRAINT
def count_teacher_at_same_time(teacher, schedules, dict_by_numbers): #Check if a teacher is scheduled in two courses at the same time
    teacher_schedules = [np.isin(schedule, [key for key, value in dict_by_numbers.items() if teacher in value]) for schedule in schedules]
    count = 0
    for i in range(len(teacher_schedules) - 1):
        for j in range(i + 1, len(teacher_schedules)):
            count += np.sum(np.logical_and(teacher_schedules[i], teacher_schedules[j]))
    return count

def count_subjects_in_schedules(schedules, dict_by_numbers): #Count if the subjects in the schedules match those specified by the user
    subjects = generate_subjects_dict(dict_by_numbers) 

    all_counts = []
    for schedule in schedules:
        subject_counts = {subject: 0 for subject in subjects.keys()}
        for row in range(len(schedule)):
            for column in range(len(schedule[row])):
                for subject, subject_number in subjects.items():
                    if schedule[row][column] in subject_number:  
                        subject_counts[subject] += 1
        all_counts.append(subject_counts)
        
    conteo2 = [np.array(list(d.values())) for d in all_counts]
        
    recuento = [abs(a - b) for a, b in zip(conteo2, hoursClass)]
    total_score = sum(recuento)
    total_score = sum(total_score)

    return total_score

#SOFT CONSTRAINTS
def calculate_repetition_score(schedules, dict_by_numbers): #Calculate if the subjects that are the same are consecutive
    subjects = generate_subjects_dict(dict_by_numbers)    
    
    # Calculate the score
    score = 0
    for schedule in schedules:
        for column in range(schedule.shape[1]):
            for subject, numbers in subjects.items():
                # Only count repetitions within the same schedule
                if np.sum(np.isin(schedule[:, column], numbers)) > 1:
                    score += 1
    return score

def teacher_overload(teachers, schedules, dict_by_numbers, max_hours_per_day):
    
    score = 0
    teacher_hours_per_day = {teacher: [0] * len(schedules[0][0]) for teacher in teachers}

    for schedule in schedules:
        for i in range(len(schedule)):
            for j in range(len(schedule[i])):
                for teacher in teachers:
                    if teacher in dict_by_numbers[schedule[i][j]]:
                        teacher_hours_per_day[teacher][j] += 1

    for teacher, hours in teacher_hours_per_day.items():
        for hour_count in hours:
            if hour_count > max_hours_per_day:
                score = -2
                return score  # Constraint violated
            score = 2
    return score  # Constraint satisfied

''' FITNESS FUNCTION '''
def fitness_function(teachers, schedules, dict_by_numbers):
        #HARD CONSTRAINTS
    hard_score1 = 0
    for teacher in teachers:  
        hard_score1 += count_teacher_at_same_time(teacher, schedules, dict_by_numbers) 
    hard_score2 = count_subjects_in_schedules(schedules, dict_by_numbers)
        #SOFT CONSTRAINTS
    soft_score1 = calculate_repetition_score(schedules, dict_by_numbers)
    soft_score2 = teacher_overload(teachers,schedules,dict_by_numbers,5)
    
    
    function = 30*hard_score1 + 30*hard_score2 - 1*soft_score1 - 1*soft_score2
    
    hard_constraints = hard_score1 + hard_score2
    if hard_constraints != 0: 
        x = 'True'
    else: 
        x = 'False'
        
    #print('H1= ',hard_score1)
    #print('H2= ',hard_score2)
    #print('S1= ',soft_score1)
    #print('S2= ',soft_score2)
    
    return function if function > 0 else 0, x 

def population_fitness_function(teachers, dict_by_numbers, population):
    all_fitness_functions = []
    for i in range(len(population)):
        value, x = fitness_function(teachers, population[i], dict_by_numbers)
        all_fitness_functions.append(value)
        
    return all_fitness_functions

''' PARENTS SELECTION '''
def sort_by_fitness(population, all_fitness_functions):
    combined = list(zip(all_fitness_functions, population))
    combined.sort(key=lambda x: x[0], reverse = True)
    
    return combined

#Ranking Selection
def ranking_selection(population, all_fitness_functions):
    index_population_sorted = sort_by_fitness(population, all_fitness_functions)
    xi, index = zip(*index_population_sorted)
    s = 2 # Have to be: 1 < s =< 2
    u = len(xi)
    probabilities = []
    if s > 1 & s <= 2:
        for i in range(len(xi)):
            value = ((2-s)/u) + ((2*i*(s-1))/(u*(u-1))) 
            #If we need it to choose only higher values, another formula must be implemented
            #(1 - e^-i)/c
            probabilities.append(value)
            
    index = list(range(len(probabilities)))
    selected_index = random.choices(index, weights=probabilities, k=2)
    
    rank_fitness_value = [xi[selected_index[0]], xi[selected_index[1]]]
    indexes = []
    for fitness_value in rank_fitness_value:
        value = [t[1] for t in index_population_sorted if t[0] == fitness_value]
        indexes.append(value[0])
    rank1 = population[indexes[0]]        
    rank2 = population[indexes[1]]
    
    return rank1, rank2      
        
#Tournament Selection
def tournament_selection(population, fitness_scores, tournament_size=3):
    # Initialize an empty list to store the randomly selected schedules
    selected_schedules = []

    # Randomly pick 'tournament_size' number of schedules
    for _ in range(tournament_size):
        random_index = random.randint(0, len(population) - 1)
        selected_schedules.append((population[random_index], fitness_scores[random_index]))
        #print(f'Se ha elegido a {f"schedules{random_index}"}, con una puntuación de {fitness_scores[random_index]}')
   
    # Sort the selected schedules by their fitness scores in descending order
    selected_schedules.sort(key=lambda x: x[1], reverse=False)  # Reverse = False --> Less score is a better score, so the first will be the smallest score
    # Select the schedule with the smallest fitness score (As smaller better is the score)
    first_best = selected_schedules[0][0]
    second_best = selected_schedules[1][0]

    return first_best, second_best

''' COMBINATION '''
def uniform_crossover(parent1, parent2):
    child1 = []
    child2 = []
    for group1, group2 in zip(parent1, parent2):
        mask = np.random.randint(0, 2, size=group1.shape).astype(bool)
        child_group1 = group1.copy()
        child_group2 = group2.copy()
        child_group1[mask] = group2[mask]
        child_group2[mask] = group1[mask]
        child1.append(child_group1)
        child2.append(child_group2)  
    return child1, child2  

def recombination(parent1, parent2, hours_per_week): 

    child_schedule = []

    for i in range(len(parent1)):

        cut_point_row = random.randint(1, hours_per_week - 1)
        cut_point_col = random.randint(1, 4)  # 4 because there are 5 days (columns)

        top_left = parent1[i][:cut_point_row, :cut_point_col]
        top_right = parent2[i][:cut_point_row, cut_point_col:]
        bottom_left = parent1[i][cut_point_row:, :cut_point_col]
        bottom_right = parent2[i][cut_point_row:, cut_point_col:]

        top_half = np.hstack((top_left, top_right))
        bottom_half = np.hstack((bottom_left, bottom_right))
        combined_schedule = np.vstack((top_half, bottom_half))
        child_schedule.append(combined_schedule)

    return child_schedule   

''' MUTATION '''
def swap_mutation(individual, drastic=False):
    for group in individual:
        row1, col1 = np.random.randint(0, group.shape[0]), np.random.randint(0, group.shape[1])
        row2, col2 = np.random.randint(0, group.shape[0]), np.random.randint(0, group.shape[1])
        
        while row1 == row2 and col1 == col2:
            row2, col2 = np.random.randint(0, group.shape[0]), np.random.randint(0, group.shape[1])

        group[row1, col1], group[row2, col2] = group[row2, col2], group[row1, col1]
        

        if drastic:
            row3, col3 = np.random.randint(0, group.shape[0]), np.random.randint(0, group.shape[1])
            row4, col4 = np.random.randint(0, group.shape[0]), np.random.randint(0, group.shape[1])

            while row3 == row4 and col3 == col4:
                row4, col4 = np.random.randint(0, group.shape[0]), np.random.randint(0, group.shape[1])

            group[row3, col3], group[row4, col4] = group[row4, col4], group[row3, col3]

''' SURVIVOR SELECTION '''
def elitist_selection(population, fitnesses, population_size):
    elite_size = population_size
    sorted_indices = np.argsort(fitnesses)
    new_population = {new_index: population[old_index] for new_index, old_index in enumerate(sorted_indices[:elite_size])}
    
    return new_population

''' PLOT SCHEDULES '''
def plot_schedule(schedule, hours_per_week, dict_by_numbers):
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    hours = list(range(8, 8 + hours_per_week))

    # Define colors for each subject ignoring professors
    unique_subjects = set([val for val in dict_by_numbers.values()])
    subject_colors = {subject: plt.cm.tab10(i) for i, subject in enumerate(unique_subjects)}
    subject_colors['---'] = '#ffff8f' #yellow for empty cells

    subjects = {key: val for key, val in dict_by_numbers.items()}
    subjects[0] = '---' #change empty cells content to ---

    for class_idx, class_schedule in enumerate(schedule):
        # Plotting the schedule with custom colors
        fig, ax = plt.subplots(figsize=(8, 6))

        # Draw rectangles for each cell
        for i in range(len(hours)):
            for j in range(len(days)):
                subject_idx = int(class_schedule[i][j])
                subject_name = subjects[subject_idx]
                color = subject_colors.get(subject_name, 'yellow')  # Default color for '---' cells
                # Draw a border rectangle for each subject cell
                rect = plt.Rectangle((j - 0.5, (len(hours) - i - 1) - 0.5), 1, 1, fill=True, edgecolor='black',
                                     facecolor=color, linewidth=2)
                ax.add_patch(rect)
                wrapped_text = textwrap.fill(subject_name, width=10)  # Adjust 'width' as needed
                plt.text(j, (len(hours) - i - 1), wrapped_text, ha='center', va='center',
                         color='black', fontweight='bold', fontsize=9)

        plt.title(f'Class {class_idx + 1} Schedule', fontweight='bold', fontsize=14)
        plt.xlabel('Days')
        plt.ylabel('Hours')
        plt.xticks(ticks=np.arange(0, len(days), 1), labels=days, ha='center')
        plt.yticks(ticks=range(len(hours)), labels=[f'{h}:00 to {h+1}:00' for h in reversed(hours)])
        ax.xaxis.tick_top()  # X-axis on top
        ax.xaxis.set_label_position('top')
        ax.set_xlim(-0.5, len(days) - 0.5)  # Set proper limits for x-axis
        ax.set_ylim(-0.5, len(hours) - 0.5)  # Set proper limits for y-axis
        ax.grid(False)  # Disable grid lines

        plt.tight_layout()
        plt.show()
        
        
if __name__ == "__main__":
    

    '''
    INITIALIZATION OF THE DATA TO CREATE THE INDIVIDUALS OF THE POPULATION
    '''
    hours_per_week = 7 #Number of work hours per week
    subjectsName = ['Math', 'Pyshics', 'Socials', 'English', 'Quemich','Science']
    hoursClass = [[5,4,4,3,4,4], #ClassA
                  [5,4,4,3,4,4], #ClassB
                  [5,4,4,3,4,4]] #ClassC
    teachers = ['Paco', 'Luis', 'Pedro', 'Juan', 'Sofia', 'Laura']
    availTime = [15,12,12,9,12,12]
    
    Teachers = assign_labels(teachers, availTime)
    #Teachers = assign_labels2(teachers, availTime) # Improve 'assign_labels'
    
    totalHours = total_subject_hours(hoursClass)
    max_hours_per_week = calculate_max_hours(hoursClass)
    
    Subjects = assign_labels(subjectsName, totalHours)
            
    
    '''
    INITIAL POBLATION GENERATION
    '''
    population_size = 20
    
    if sum(totalHours) <= sum(availTime):
        Index = assign_localization(Subjects, Teachers)
        
        if  hours_per_week*5 >= max_hours_per_week:
            dict_by_strings, dict_by_numbers = assign_dictionaries(Index)
            
            #population =  generate_population(population_size, hoursClass, hours_per_week, totalHours, Index, dict_by_strings)
            population = generate_mix_population(population_size, hoursClass, hours_per_week, totalHours, Index, dict_by_strings)
        else: 
            sys.exit('Error in the data, INCREASE the availability of hours per week')
            
    else:
        sys.exit('Error in the input data, INCREASE the availability of hours per teacher')
        
        
    '''
    BEGINNING OF THE ALGORITHM
    '''
    best_score = 1000
    generation = 0
    max_generations = 1000
    fitness_goal = 0
    mutation_rate = 0.4
    
    while generation < max_generations:
        
        '''
        1. FITNESS FUNCTION CALCULATION
        '''
        totalfitness = population_fitness_function(teachers, dict_by_numbers, population)

        '''
        2. PARENTS SELECTION
        '''
        # parent1, parent2 = tournament_selection(population, totalfitness)
        
        # Improve to choose fathers
        parent1, parent2 = ranking_selection(population, totalfitness)
        
        '''
        3. GENERATION OF DESCENDANTS
        '''
        child1, child2 = uniform_crossover(parent1, parent2)
        #child1 = recombination(parent1, parent2, hours_per_week)
        #child2 = recombination(parent1, parent2, hours_per_week)

        
        '''
        4. MUTATION
        '''

        if random.random() < mutation_rate:
            swap_mutation(child1)
            swap_mutation(child2)
        
        '''
        5. AGGREGATION OF CHILDREN TO THE POPULATION
        '''
        population[len(population)] = child1
        population[len(population)] = child2

        '''
        6. SURVIVORS SELECTION
        '''
        totalfitness = population_fitness_function(teachers, dict_by_numbers, population)
        population = elitist_selection(population, totalfitness, population_size)
        
        '''
        7. EXTRACTING THE BEST PROSPECT OF THE NEW POPULATION'''
        best_prospect = choose_best_prospect(teachers, dict_by_numbers, population)
        best_score, have_constraints = fitness_function(teachers, best_prospect, dict_by_numbers)
        #print(totalfitness)
        print('Fitness Function of the best schedule: ',best_score)
        print('Have hard constraints? ',have_constraints)
        print('Iteration: ',generation)
        generation +=1
        if best_score < fitness_goal or have_constraints == 'False': break

    if(best_score >= 10):
        print("More iterations are needed to fins an optimal schedule that meet the hard constrains")
    print(best_prospect)
    plot_schedule(best_prospect, hours_per_week, dict_by_numbers)
    


    
    
    
    
        

    
    
    
    

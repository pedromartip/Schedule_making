import random
import copy
import numpy as np
import matplotlib.pyplot as plt

def generate_schedule(num_classes, subjects, teachers, hours_per_subject, teacher_assignment):
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    hours = list(range(8, 18))

    # Initialize schedules
    schedules = []


    for class_num in range(1, num_classes + 1):
        schedule = {}
        subject_hours = hours_per_subject.copy()  # Copy of required hours for subjects

        for day in days:
            for hour in hours:
                subject = random.choice(subjects)

                # Check if teacher is available and if subject hours are not exceeded
                if teacher_assignment[subject] in teachers and subject_hours[subject] > 0:
                    schedule[(day, hour)] = subject
                    subject_hours[subject] -= 1
                else: 
                    schedule[(day, hour)] = '---'

        schedules.append(schedule)

    return schedules

def generate_poblation(poblation, num_classes, subjects, teachers, hours_per_subject, teacher_assignment):
    # Inicializar el diccionario de horarios
    schedules = {}

    for i in range(poblation):
        # Generar un horario
        schedule = generate_schedule(num_classes, subjects, teachers, hours_per_subject, teacher_assignment)
        
        # Agregar el horario al diccionario de horarios
        schedules[f'schedules{i+1}'] = schedule

    return schedules

def print_schedule(schedule):
    # Print the schedule in a readable format
    print("{:<10} {:<10} {:<10} {:<10} {:<10} {:<10}".format('', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'))
    for hour in range(8, 18):
        print("{:<10}".format(f'{hour}:00-{hour + 1}:00'), end=' ')
        for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
            print("{:<10}".format(schedule.get((day, hour), '')), end=' ')
        print()

def plot_schedule(schedule):
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    hours = list(range(8, 18))

    # Create a 2D array to represent the schedule and initialize with empty strings
    schedule_array = [['' for _ in range(len(days))] for _ in range(len(hours))]

    # Fill the schedule array with subject indices
    subjects = list(set(schedule.values()))
    subject_indices = {subject: i for i, subject in enumerate(subjects)}

    # Define colors for each subject
    subject_colors = {
        'Math': 'cyan',
        'Physics': 'limegreen',
        'History': 'red',
        'Chemistry': 'orange',
        'English': 'magenta'
        # Add or change more subjects and colors as needed
    }

    for i, day in enumerate(days):
        for j, hour in enumerate(hours):
            subject = schedule.get((day, hour), '---')
            schedule_array[j][i] = subject_indices[subject]

    # Create a matrix with integers representing subject indices
    schedule_matrix = np.array(schedule_array, dtype=int)

    # Create a list of colors corresponding to each subject index
    colors = [subject_colors[subjects[i]] if subjects[i] in subject_colors else 'yellow' for i in range(len(subjects))]

    # Plotting the schedule with custom colors
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(schedule_matrix, cmap='viridis', aspect='auto', vmin=0, vmax=len(subjects) - 1)

    # Draw rectangles for each cell
    for i in range(len(hours)):
        for j in range(len(days)):
            subject_idx = schedule_matrix[i][j]
            color = colors[subject_idx] if subject_idx != -1 else 'yellow'
            # Draw a border rectangle for each subject cell
            rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=True, edgecolor='black', facecolor=color, linewidth=2)
            ax.add_patch(rect)
            plt.text(j, i, subjects[subject_idx] if subject_idx != -1 else 'Empty', ha='center', va='center', color='black', fontweight='bold', fontsize=12)

    plt.title(f'Class {class_num} Schedule',fontweight='bold', fontsize=14)
    plt.xlabel('Days')
    plt.ylabel('Hours')
    plt.xticks(ticks=range(len(days)), labels=days)
    plt.yticks(ticks=range(len(hours)), labels=[f'{h}:00' for h in hours])
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.grid(False)  # Disable grid lines

    plt.show()

"""
Here we will calculate the fitness score throught the hard and soft constraints
"""
def calculate_fitness(member, hours_per_subject, teacher_assignment, teacher_max_hours):
    score = 170
    hard_constraint = 50 
    soft_constraint = 10 

    member_schedules = {}
    for schedule in member:
        for key, value in schedule.items():
            if key in member_schedules:
                member_schedules[key].append(value)
            else:
                member_schedules[key] = [value]

    # Hard Constraints
    if has_overlapping_classes(member_schedules, teacher_assignment):
        score -= hard_constraint
        #print(f'Hard constraint violada, new puntuation = {score}')

    if not meets_required_hours(member, hours_per_subject):
        score -= hard_constraint
        #print(f'Hard constraint violada, new puntuation = {score}')

    if exceeds_teacher_max_hours(member_schedules, teacher_assignment, teacher_max_hours):
        score -= hard_constraint
        #print(f'Hard constraint violada, new puntuation = {score}')

    # Soft Constraints
    if not evenly_distributed_subjects(schedule):
        score -= soft_constraint
        #print(f'Soft constraint violada, new puntuation = {score}')

    if not minimized_idle_hours(schedule, teacher_assignment):
        score -= soft_constraint
        #print(f'Soft constraint violada, new puntuation = {score}')

    return score

"""
HARD CONSTRAINTS
"""
# This function check if a teacher is doing more hours reather than the assigned ones
def exceeds_teacher_max_hours(member_schedules, teacher_assignment, teacher_max_hours):
    # Initialize a dictionary to count the scheduled hours for each teacher
    assigned_hours = {teacher: 0 for teacher in teacher_max_hours}

    # Iterate over the schedule to count the hours for each teacher
    for subjects in member_schedules.values():
        for subject in subjects:
            try:
                teacher = teacher_assignment[subject]
                assigned_hours[teacher] += 1
            except:
                pass

    # Check if any teacher exceeds their maximum hours
    for teacher, max_hours in teacher_max_hours.items():
        if assigned_hours[teacher] > max_hours:
            return True  # This teacher has exceeded their maximum hours

    return False  # No teacher has exceeded their maximum hours


# This function check if any teacher is scheduled for two subjects at a same time
def has_overlapping_classes(member_schedules, teacher_assignment):
    teacher_timetable = {teacher: [] for teacher in teacher_assignment.values()}
    
    for (day, hour), subjects in member_schedules.items():
        for subject in subjects:
            try: 
                teacher = teacher_assignment[subject]
                if (day, hour) in teacher_timetable[teacher]:
                    return True
                teacher_timetable[teacher].append((day, hour))
            except:
                pass

    return False


# This function verifies if the schedulematch with the requiered hours for each subject
def meets_required_hours(member_schedules, hours_per_subject):

    for schedule in member_schedules:
        scheduled_hours = {subject: 0 for subject in hours_per_subject}

        for subjects in schedule.items():
            try:
                if subject in scheduled_hours:
                    scheduled_hours[subject] += 1
            except: 
                pass

        for subject, required_hours in hours_per_subject.items():
            if scheduled_hours[subject] != required_hours:
                return False

    return True

"""
SOFT CONSTRAINTS
"""
# This function check if the subjects are well spread trhought the week
# --> We don't want to do one subject only in one day, for example
def evenly_distributed_subjects(schedule):
    daily_subject_count = {day: {} for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']}

    for (day, _), subject in schedule.items():
        if subject in daily_subject_count[day]:
            daily_subject_count[day][subject] += 1
        else:
            daily_subject_count[day][subject] = 1

    for day, subjects in daily_subject_count.items():
        if len(subjects) > 1 and max(subjects.values()) - min(subjects.values()) > 1:
            return False

    return True

# This functions pay attention to the gaps in each teacher's schedule
def minimized_idle_hours(schedule, teacher_assignment):
    teacher_idle_hours = {teacher: 0 for teacher in teacher_assignment.values()}

    for teacher in teacher_idle_hours:
        daily_schedule = {day: [] for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']}
        
        for (day, hour), subject in schedule.items():
            try:
                if teacher_assignment[subject] == teacher:
                    daily_schedule[day].append(hour)
            except:
                pass

        for hours in daily_schedule.values():
            sorted_hours = sorted(hours)
            idle_periods = [sorted_hours[i+1] - sorted_hours[i] - 1 for i in range(len(sorted_hours)-1)]
            teacher_idle_hours[teacher] += sum(idle_periods)

    return sum(teacher_idle_hours.values()) == 0

"""
TOURNAMENT SELECTION
"""
def tournament_selection(population, fitness_scores, tournament_size=3):
    # Initialize an empty list to store the randomly selected schedules
    selected_schedules = []
    #print(f'population: {len(population)}')
    print('EMPIEZA EL TORNEO..')
    # Randomly pick 'tournament_size' number of schedules
    for _ in range(tournament_size):
        random_index = random.randint(0, len(population) - 1)
        selected_schedules.append((population[f'schedules{random_index+1}'], fitness_scores[random_index]))
        print(f'Se ha elegido a {f"schedules{random_index+1}"}, con una puntuación de {fitness_scores[random_index]}')
    # Sort the selected schedules by their fitness scores in descending order
    selected_schedules.sort(key=lambda x: x[1], reverse=True)

    # Select the schedule with the highest fitness score
    best_schedule = selected_schedules[0][0]

    return best_schedule

"""
RECOMBINATION AND MUTATION
"""

"""
recombine_population
poblation1, poblation2 --> El primero y el segundo ganador
"""
def recombine_population(poblation1, poblation2):
    # Asegúrate de que ambas poblaciones tienen el mismo tamaño
    assert len(poblation1) == len(poblation2)
    
    new_poblation1 = []
    new_poblation2 = []
    
    # Aplica la función recombine a cada par de diccionarios
    for schedule1, schedule2 in zip(poblation1, poblation2):
        
        # Obtén todas las horas posibles en el horario
        times = list(schedule1.keys())
        
        # Selecciona un punto de corte aleatorio
        cut_point = random.randint(0, len(times))
        
        # Crea los nuevos horarios
        new_schedule1 = {}
        new_schedule2 = {}
        
        # Intercambia las partes del horario después del punto de corte
        for i, time in enumerate(times):
            if i < cut_point:
                new_schedule1[time] = schedule1[time]
                new_schedule2[time] = schedule2[time]
            else:
                new_schedule1[time] = schedule2[time]
                new_schedule2[time] = schedule1[time]
        
        new_poblation1.append(new_schedule1)
        new_poblation2.append(new_schedule2)
    
    return new_poblation1, new_poblation2

'''
Es otro tipo de recombinacion
'''
def recombination(poblation1, poblation2, hours_per_subject):
    assert len(poblation1) == len(poblation2), "Los padres deben tener el mismo número de genes"
    
    new_poblation1 = []
    new_poblation2 = []
    
    for padre1, padre2 in zip(poblation1, poblation2):
        hijo1, hijo2 = {}, {}
        materias = list(set(padre1.values()).union(set(padre2.values())))
        
        for materia in materias:
            if materia != '---':
                # Obtener los genes donde la materia es impartida en cada padre
                genes_padre1 = [gene for gene, valor in padre1.items() if valor == materia]
                genes_padre2 = [gene for gene, valor in padre2.items() if valor == materia]
                
                # Mezclar los genes
                random.shuffle(genes_padre1)
                random.shuffle(genes_padre2)
                
                # Obtener la cantidad de horas que se deben asignar para esta materia
                horas = hours_per_subject[materia]
                
                # Asignar las horas a los hijos
                for i in range(min(horas, len(genes_padre1), len(genes_padre2))):
                    if i % 2 == 0:
                        hijo1[genes_padre1[i]] = materia
                        hijo2[genes_padre2[i]] = materia
                    else:
                        hijo1[genes_padre2[i]] = materia
                        hijo2[genes_padre1[i]] = materia
        
        # Rellenar los genes restantes con '---'
        for gene in padre1.keys():
            if gene not in hijo1:
                hijo1[gene] = '---'
            if gene not in hijo2:
                hijo2[gene] = '---'
                
        new_poblation1.append(hijo1)
        new_poblation2.append(hijo2)

    return new_poblation1, new_poblation2


def mutation(scheduleOriginal):
    '''
    Se puede hacer que mute un unico horario cuando estemos cerca de la respuesta, para una 
    una diversidad controlada, pero si estamos lejos de la respuesta podriamos hacer mutar a todos 
    los horarios dentro de schedules para una diversidad mayor y explorar mas posibilidades
    '''
    schedules = copy.deepcopy(scheduleOriginal)
    
    rand = random.choice(range(len(schedules)))
    schedule = schedules[rand]

    # Obtén todos los días y horas posibles en el horario
    days_hours = list(schedule.keys())

    # Selecciona dos días y horas aleatorios
    day_hour1 = random.choice(days_hours)
    day_hour2 = random.choice(days_hours)

    # Intercambia las asignaturas programadas para esos días y horas
    while schedule[day_hour1] == '---' and schedule[day_hour2] == '---' or schedule[day_hour1] == schedule[day_hour2]:
      day_hour1 = random.choice(days_hours)
      day_hour2 = random.choice(days_hours)  

    schedule[day_hour1], schedule[day_hour2] = schedule[day_hour2], schedule[day_hour1]

    schedules[rand] = schedule
    
    
    return schedules


def teacherAssignment(subjects, teachers):
    # Assign teachers to subjects
    if len(subjects) == len(teachers):
        teacher_assignment = {subjects[i]: teachers[i] for i in range(len(subjects))}
    else:
        teacher_assignment = {subject: random.choice(teachers) for subject in subjects}
        
    return teacher_assignment


#Podemos agregar un codigo que permita solo 2 horas seguidas de la misma materia, para que el horario sea mejor 
#podria ser un soft constraint

if __name__ == "__main__":
    num_classes = 3 # Number of classes for which the schedule needs to be generated.
    
    #Declare inputs
    teacher_max_hours = {'Pep': 20, 'Juan': 20, 'Cr7':20 , 'Puigdemont': 20, 'Francisco F.': 20}
    hours_per_subject = {'Math': 6, 'English': 9, 'Chemistry': 4, 'History': 5, 'Physics': 6}
    subjects = list(hours_per_subject.keys())
    teachers = list(teacher_max_hours.keys())
    teacher_assignment = teacherAssignment(subjects, teachers)
    
    #Initial poblation
    num_poblation = 20
    poblation = generate_poblation(num_poblation, num_classes, subjects, teachers, hours_per_subject, teacher_assignment)
    max_puntuacion = 0
    max_generations = 3
    fitness_goal = 150
    
    print("START\n")
    print('-------------------------------------------------')
    print(f'CREAMOS LA PRIMERA GENERACIÓN CON POBLACIÓN = {num_poblation}\n')

    for class_num, schedule in enumerate(poblation['schedules1'], start=1):
       print(f"\nSchedule for Class {class_num}:\n")
       print_schedule(schedule)
       print('\n' + '-'*50)  # Separate shcedules with a line for better visualisation

    for generation in range(max_generations): # Si se llega a mas de 50 (51 o más, quiere decir que acepta las hard constrains)
        print('-------------------------------------------------')
        print('NEW ITERATION')
        # Calculamos la fitness score de cada miembro de la población
        puntuaciones = []
        for member in poblation:
            #for schedule in poblation[member]:
            # print(f'Miramos => {member}')
            fitness_points = calculate_fitness(poblation[member],hours_per_subject,teacher_assignment,teacher_max_hours)
            puntuaciones.append(fitness_points)

        
        # encontramos los índices de los miembros con peor puntuación
        worst_fitness = sorted(range(len(poblation)), key=lambda i: puntuaciones[i])[:2]

        # Tournament
        print("---TOURNAMENT---")
        padre1 = tournament_selection(poblation, puntuaciones)
        padre2 = tournament_selection(poblation, puntuaciones)

        hijo1, hijo2 = recombination(padre1, padre2, hours_per_subject)

        hijo1 = mutation(hijo1)
        hijo2 = mutation(hijo2)

        # Reemplazamos los peores mimebros
        for i in worst_fitness:    
            poblation[f'schedules{i+1}'] = hijo1 if i == 0 else hijo2
            
        max_index = puntuaciones.index(max(puntuaciones))
        best_schedule = poblation[f'schedules{max_index+1}']
        max_puntuacion = puntuaciones[max_index]

        print('Mejor schedule:')
        for class_num, schedule in enumerate(best_schedule, start=1):
            print(f"\nSchedule for Class {class_num}:\n")
            print_schedule(schedule)
            # plot_schedule(schedule)
            print('\n' + '-'*50)  # Separate shcedules with a line for better visualisation

        print("\n-----------------------------------------------------\n")
        print(f'Generación #: {generation} , Mejor puntuacion = {max_puntuacion}')
        
        if max_puntuacion > fitness_goal:
            break       
         
           
        
        

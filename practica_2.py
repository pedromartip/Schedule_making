import random
import copy

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

"""
Here we will calculate the fitness score throught the hard and soft constraints
"""
def calculate_fitness(schedule, hours_per_subject, teacher_assignment, teacher_max_hours):
    score = 0
    hard_constraint = 50 
    soft_constraint = 10 

    # Hard Constraints
    if has_overlapping_classes(schedule, teacher_assignment):
        score -= hard_constraint

    if not meets_required_hours(schedule, hours_per_subject):
        score -= hard_constraint

    if exceeds_teacher_max_hours(schedule, teacher_assignment, teacher_max_hours):
        score -= hard_constraint

    # Soft Constraints
    if not evenly_distributed_subjects(schedule):
        score -= soft_constraint

    if not minimized_idle_hours(schedule, teacher_assignment):
        score -= soft_constraint

    return score

"""
HARD CONSTRAINTS
"""
# This function check if a teacher is doing more hours reather than the assigned ones
def exceeds_teacher_max_hours(schedule, teacher_assignment, teacher_max_hours):
    # Initialize a dictionary to count the scheduled hours for each teacher
    scheduled_hours = {teacher: 0 for teacher in teacher_max_hours}

    # Iterate over the schedule to count the hours for each teacher
    for _, subject in schedule.items():
        try:
            teacher = teacher_assignment[subject]
            scheduled_hours[teacher] += 1
        except:
            pass

    # Check if any teacher exceeds their maximum hours
    for teacher, max_hours in teacher_max_hours.items():
        if scheduled_hours[teacher] > max_hours:
            return True  # This teacher has exceeded their maximum hours

    return False  # No teacher has exceeded their maximum hours


# This function check if any teacher is scheduled for two subjects at a same time
def has_overlapping_classes(schedule, teacher_assignment):
    teacher_timetable = {teacher: [] for teacher in teacher_assignment.values()}
    
    for (day, hour), subject in schedule.items():
        try: 
            teacher = teacher_assignment[subject]
            if (day, hour) in teacher_timetable[teacher]:
                return True
            teacher_timetable[teacher].append((day, hour))
        except:
            pass

    return False


# This function verifies if the schedulematch with the requiered hours for each subject
def meets_required_hours(schedule, hours_per_subject):
    scheduled_hours = {subject: 0 for subject in hours_per_subject}

    try:
        for subject in schedule.values():
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
RECOMBINATION AND MUTATION
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
def recombine_uniform(poblation1, poblation2):
    # Asegúrate de que ambas poblaciones tienen el mismo tamaño
    assert len(poblation1) == len(poblation2)
    
    new_poblation1 = []
    new_poblation2 = []
    
    for schedule1, schedule2 in zip(poblation1, poblation2):
        # Obtén todas las horas posibles en el horario
        times = list(schedule1.keys())
        
        # Crea los nuevos horarios
        new_schedule1 = {}
        new_schedule2 = {}
        
        # Para cada hora, selecciona aleatoriamente la asignatura de uno de los horarios parentales
        for time in times:
            if random.random() < 0.5:
                new_schedule1[time] = schedule1[time]
                new_schedule2[time] = schedule2[time]
            else:
                new_schedule1[time] = schedule2[time]
                new_schedule2[time] = schedule1[time]
                
        new_poblation1.append(new_schedule1)
        new_poblation2.append(new_schedule2)
    
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
    num_classes = 1 # Number of classes for which the schedule needs to be generated.
    
    #Declare inputs
    teacher_max_hours = {'Pep': 20, 'Juan': 20, 'Cr7':20 , 'Puigdemont': 20, 'Francisco F.': 20}
    hours_per_subject = {'Math': 1, 'English': 1, 'Chemistry': 1, 'History': 1, 'Physics': 1}
    subjects = list(hours_per_subject.keys())
    teachers = list(teacher_max_hours.keys())
    teacher_assignment = teacherAssignment(subjects, teachers)
    
    num_poblation = 5
    
    poblation = generate_poblation(num_poblation, num_classes, subjects, teachers, hours_per_subject, teacher_assignment)
    
    
    hijo1, hijo2 = recombine_population(poblation['schedules1'], poblation['schedules2'])
    #hijo1, hijo2 = recombine_uniform(poblation['schedules1'], poblation['schedules2'])
    
    #mutado = mutation(hijo2)
    
    #fitness_function = calculate_fitness(schedules[1], hours_per_subject, teacher_assignment, teacher_max_hours)
    
           
    for class_num, schedule in enumerate(poblation['schedules1'], start=1):
       print(f"\nSchedule for Class {class_num}:\n")
       print_schedule(schedule)
       print('\n' + '-'*50)  # Separate shcedules with a line for better visualisation
       
    for class_num, schedule in enumerate(poblation['schedules2'], start=1):
       print(f"\nSchedule for Class {class_num}:\n")
       print_schedule(schedule)
       print('\n' + '-'*50)  # Separate shcedules with a line for better visualisation
           
    for class_num, schedule in enumerate(hijo1, start=1):
       print(f"\nSchedule for Class {class_num}:\n")
       print_schedule(schedule)
       print('\n' + '-'*50)  # Separate shcedules with a line for better visualisation
       
    for class_num, schedule in enumerate(hijo2, start=1):
       print(f"\nSchedule for Class {class_num}:\n")
       print_schedule(schedule)
       print('\n' + '-'*50)  # Separate shcedules with a line for better visualisation
         
           
        
        

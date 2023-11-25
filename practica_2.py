import random

def generate_schedule(num_classes, subjects, teachers, hours_per_subject):
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    hours = list(range(8, 18))

    # Initialize schedules
    schedules = []

    # Assign teachers to subjects
    teacher_assignment = {subject: random.choice(teachers) for subject in subjects}

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
        fitness_score -= hard_constraint

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
        teacher = teacher_assignment[subject]
        scheduled_hours[teacher] += 1

    # Check if any teacher exceeds their maximum hours
    for teacher, max_hours in teacher_max_hours.items():
        if scheduled_hours[teacher] > max_hours:
            return True  # This teacher has exceeded their maximum hours

    return False  # No teacher has exceeded their maximum hours


# This function check if any teacher is scheduled for two subjects at a same time
def has_overlapping_classes(schedule, teacher_assignment):
    teacher_timetable = {teacher: [] for teacher in teacher_assignment.values()}
    
    for (day, hour), subject in schedule.items():
        teacher = teacher_assignment[subject]
        if (day, hour) in teacher_timetable[teacher]:
            return True
        teacher_timetable[teacher].append((day, hour))

    return False


# This function verifies if the schedulematch with the requiered hours for each subject
def meets_required_hours(schedule, hours_per_subject):
    scheduled_hours = {subject: 0 for subject in hours_per_subject}

    for subject in schedule.values():
        scheduled_hours[subject] += 1

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
            if teacher_assignment[subject] == teacher:
                daily_schedule[day].append(hour)

        for hours in daily_schedule.values():
            sorted_hours = sorted(hours)
            idle_periods = [sorted_hours[i+1] - sorted_hours[i] - 1 for i in range(len(sorted_hours)-1)]
            teacher_idle_hours[teacher] += sum(idle_periods)

    return sum(teacher_idle_hours.values()) == 0

"""
RECOMBINATION AND MUTATION
"""

def recombination(parent1, parent2):
    children1 = dict(parent1)
    children2 = dict(parent2)

    # Select the cross point
    cross_point = len(children1) // 2

    # Swap subjects between parents
    parent1_keys = list(parent1.keys())
    for i in range(cross_point, len(children1)):
        children1[parent1_keys[i]], children2[parent1_keys[i]] = children2[parent1_keys[i]], children1[parent1_keys[i]]

    return children1, children2


def mutacion(schedule, subjects):
    muted_schedule = dict(schedule)

    # Select a random position to make the mutation
    mutation_index = random.choice(range(len(muted_schedule)))

    # Change the subject in the selected position
    mutation_site = list(muted_schedule.keys())[mutation_index]
    new_subject = random.choice(subjects)  
    muted_schedule[mutation_site] = new_subject

    return muted_schedule


#Podemos agregar un codigo que permita solo 2 horas seguidas de la misma materia, para que el horario sea mejor 
#podria ser un soft constraint

if __name__ == "__main__":
    num_classes = 5 # Number of classes for which the schedule needs to be generated.
    subjects = ['Math', 'English', 'Chemistry', 'History', 'Physics']
    teachers = ['Pep', 'Juan', 'Cr7', 'Puigdemont', 'Francisco F.']
    hours_per_subject = {'Math': 6, 'English': 6, 'Chemistry': 6, 'History': 6, 'Physics': 6}

    schedules = generate_schedule(num_classes, subjects, teachers, hours_per_subject)
    
    #hijo1, hijo2 = recombination(schedules[1],schedules[2])
    
    for class_num, schedule in enumerate(schedules, start=1):
        print(f"\nSchedule for Class {class_num}:\n")
        print_schedule(schedule)
        print('\n' + '-'*50)  # Separate shcedules with a line for better visualisation
        
           
        
        

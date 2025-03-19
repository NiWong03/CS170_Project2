import random as rand
import math
# data was being read as a string, converted to arrays
def read_data(file_path):
    with open(file_path, 'r') as file:
        data = []
        for line in file:
            data.append([float(x) for x in line.split()])
        return data

small_data = read_data('CS170_Small_Data__111.txt')
large_data = read_data('CS170_Large_Data__12.txt')

current_set = []
data = small_data

def leave_one_out_cross_validation(data, current_set,feature_to_add):
    features = current_set + [feature_to_add]
    copy_data = [row[:] for row in data] 
# clean data, set unused to 0
    for j in range(1,len(copy_data[0])):
        if j not in features:
            for i in range(len(copy_data)):
                copy_data[i][j] = 0
                
    # print("features:", features)

    num_classified_correctly = 0
    for i in range(len(copy_data)):
        object_to_classify = copy_data[i][1:]
        label_object_to_classify = copy_data[i][0]
        # print(f"Looping over i, at the {i} location")
        # print(f"The {i}th object is in the class {label_object_to_classify}")
        nearest_neighbor_distance = float('inf')
        nearest_neighbor_location = float('inf')
        for j in range(1,len(copy_data)):
            if i != j:
                # print(f"Ask if {i} is nearest neighbor with {j}")
                distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(object_to_classify, copy_data[j][1:])))
                if distance < nearest_neighbor_distance:
                    nearest_neighbor_distance = distance
                    nearest_neighbor_location = j
                    nearest_neighbor_label = copy_data[nearest_neighbor_location][0]
            # print(f"Object {i} is in class {label_object_to_classify}")
            # print(f"Its nearest neighbor is {nearest_neighbor_location} which is in class {nearest_neighbor_label} ")
        if label_object_to_classify == nearest_neighbor_label:
            num_classified_correctly += 1
    print(f"Accuracy: {num_classified_correctly} {len(copy_data)}")
    accuracy = num_classified_correctly / len(copy_data)

        # print(f"Object {i} is in class {label_object_to_classify}")
        # print(f"Its nearest neighbor is {nearest_neighbor_location} which is in class {nearest_neighbor_label} ")

    
    # accuracy = rand.random()
    return accuracy


def forward_feature_selection(data):
    print("Beginning Forward Selection Algorithm")
    current_set = []
    best_feature = []
    best_accuracy = 0
    
    for i in range(1, len(data[0])):
        print(f'On the {i}th level of the search tree')
        best_so_far_accuracy = 0
        add_feature = None  # initialize add_feature to None

        for j in range(1, len(data[0])):
            if j not in current_set:  
                print(f'Considering adding the {j}th feature')
                accuracy = leave_one_out_cross_validation(data, current_set, j)
                print(f'-- Considering features {current_set + [j]} with accuracy of {accuracy}')
                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    print(f'Accuracy is {accuracy} and best_so_far accuracy is {best_so_far_accuracy}')
                    add_feature = j

        if add_feature is not None:
            current_set.append(add_feature)
            print(f'Feature Set {current_set} was best, accuracy of {best_so_far_accuracy}')

        if best_so_far_accuracy > best_accuracy:
            best_accuracy = best_so_far_accuracy
            best_feature = current_set[:]

    print(f'Finished Search!! The best subset is {best_feature} with an accuracy of {best_accuracy}')

def backward_feature_selection(data):
    print("Beginning Backward Elimination Algorithm")
    current_set = list(range(1, len(data[0])))  # Start with all features
    best_feature = current_set[:]
    best_accuracy = leave_one_out_cross_validation(data, current_set, 0)  # Initial accuracy with all features
    
    print(f'Considering all features {current_set} with accuracy of {best_accuracy}')
    
    for i in range(1, len(data[0]) ):
        print(f'On the {i}th level of the search tree')
        best_so_far_accuracy = 0
        remove_feature = None  # Initialize remove_feature to None

        for j in range(1, len(data[0]) - 1):
            if j in current_set:  
                print(f'Considering removing the {j}th feature')
                temp_set = current_set[:]
                temp_set.remove(j)
                # make copy to evaluate new set with removed feature
                accuracy = leave_one_out_cross_validation(data, temp_set, None)
                #pass in new set, None since deleting features not adding
                print(f'-- Considering features {temp_set} with accuracy of {accuracy}')
                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    print(f'Accuracy is {accuracy} and best_so_far accuracy is {best_so_far_accuracy}')
                    remove_feature = j

        if remove_feature is not None:
            current_set.remove(remove_feature)
            print(f'Feature {remove_feature} is irrelevant, accuracy of {best_so_far_accuracy}')

        if best_so_far_accuracy > best_accuracy:
            best_accuracy = best_so_far_accuracy
            best_feature = current_set[:]

    print(f'Finished search!! The best feature subset is {best_feature} with an accuracy of {best_accuracy}')


forward_feature_selection(data)
# backward_feature_selection(data)


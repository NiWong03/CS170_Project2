# import random as rand
import math

current_set = []


def leave_one_out_cross_validation(data, current_set,feature_to_add):
    features = current_set + [feature_to_add]
    copy_data = [row[:] for row in data] 
# clean data, set unused to 0
    for j in range(1,len(copy_data[0])):
        if j not in features:
            for i in range(len(copy_data)):
                copy_data[i][j] = 0
                


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
    accuracy = num_classified_correctly / len(copy_data)
    
    # accuracy = rand.random()
    return accuracy


def forward_feature_selection(data):
    print("Beginning Forward Selection Algorithm")
    current_set = []
    best_feature = []
    best_accuracy = 0
    
    for i in range(1, len(data[0])):
        # print(f'On the {i}th level of the search tree')
        best_so_far_accuracy = 0
        add_feature = None  # initialize add_feature to None

        for j in range(1, len(data[0])):
            if j not in current_set:  
                # print(f'Considering adding the {j}th feature')
                accuracy = leave_one_out_cross_validation(data, current_set, j)
                # print(f'-- Considering features {current_set + [j]} with accuracy of {accuracy}')
                feature_set_str = '{' + ', '.join(map(str, current_set + [j])) + '}'
                # turn set into string in order to get curly brackets for output
                print('Using Feature(s) ' + feature_set_str + ' with accuracy of ' + str(accuracy))
                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    add_feature = j

        if add_feature is not None:
            current_set.append(add_feature)
            feature_set_str = '{' + ', '.join(map(str, current_set)) + '}'
            print('Feature Set ' + feature_set_str + ' with accuracy of ' + str(best_so_far_accuracy))


        if best_so_far_accuracy > best_accuracy:
            best_accuracy = best_so_far_accuracy
            best_feature = current_set[:]
        
        best_set_str = '{' + ', '.join(map(str, best_feature)) + '}'
    print('Finished Search!! The best feature subset is ' + best_set_str + ' which has an accuracy of ' + str(best_accuracy))

def backward_feature_selection(data):
    print("Beginning Backward Elimination Algorithm")
    current_set = list(range(1, len(data[0])))  # Start with all features
    best_feature = current_set[:]
    best_accuracy = leave_one_out_cross_validation(data, current_set, 0)  # initial accuracy with all features
    feature_set_str = '{' + ', '.join(map(str, current_set )) + '}'

    print('Considering all features ' + feature_set_str + ' with accuracy of ' + str(best_accuracy))
    
    for i in range(1, len(data[0]) ):
        # print(f'On the {i}th level of the search tree')
        best_so_far_accuracy = 0
        remove_feature = None  # initialize remove_feature to None

        for j in range(1, len(data[0])-1):
            if j in current_set:  
                # print(f'Considering removing the {j}th feature')
                temp_set = current_set[:]
                temp_set.remove(j)
                # make copy to evaluate new set with removed feature
                accuracy = leave_one_out_cross_validation(data, temp_set, None)
                #pass in new set, None since deleting features not adding
                feature_set_str = '{' + ', '.join(map(str, current_set )) + '}'
                print('Using Feature(s) ' + feature_set_str + ' with accuracy of ' + str(accuracy))

                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    remove_feature = j

        if remove_feature is not None:
            current_set.remove(remove_feature)
            print(f'Feature {remove_feature} is irrelevant, accuracy of {best_so_far_accuracy}')

        if best_so_far_accuracy > best_accuracy:
            best_accuracy = best_so_far_accuracy
            best_feature = current_set[:]

    best_set_str = '{' + ', '.join(map(str, best_feature )) + '}'
    print('Finished search!! The best feature subset is ' + best_set_str + ' which has an accuracy of' + str(best_accuracy))


# data was being read as a string, converted to arrays
def read_data(file_path):
    with open(file_path, 'r') as file:
        data = []
        for line in file:
            data.append([float(x) for x in line.split()])
        return data

small_data = read_data('CS170_Small_Data__111.txt')
large_data = read_data('CS170_Large_Data__25.txt')


def main():
    print("Welcome to my Feature Selection algorithm.")
    userint = input("Type in the name of the file to test: ")

    if userint == "CS170_Large_Data__25.txt":
        data = large_data
    else:
        data = small_data
    userint = input("Type the number of the algorithm you want to run. \n 1) Forward Selection \n 2) Backward Elimination \n")
    
    if userint == "1":
        forward_feature_selection(data)   
    else:
        backward_feature_selection(data)

if __name__ == "__main__":
    main()


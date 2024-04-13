'''
Usage: modify the filename variable to the name of the jsonl file you want to analyze
'''


import json

# loads the data from a jsonl file into a list of dictionaries
def load_data(filename):
    data_list = []

    with open(filename, 'r') as f:
        for line in f:
            data = json.loads(line)
            data_list.append(data)
    return data_list


def main(): 
    correct = []
    incorrect = []
    decision = []
    correctly_accepted = []
    correctly_not_accepted = []

    filename = "data.jsonl"
    data = load_data(filename)

    for d in data:
        username = d['username']
        ground_truth = d['gt-label']
        ai_prediction = d['ai-prediction']['label']
        accepted = d['decision']
        decision.append(accepted)
        print(f"Ground Truth: {ground_truth}\tAI Prediction: {ai_prediction}\tAccepted: {accepted}")
    
        if ground_truth in ai_prediction:
            correct.append(ground_truth)
            if accepted == "Accept":
                correctly_accepted.append(ground_truth)
    else:
            incorrect.append(ground_truth) 
            if accepted == "Reject":
                correctly_not_accepted.append(ground_truth)



    print(f'\nNumber of correctly accepted or rejected: {len(correctly_accepted) + len(correctly_not_accepted)} out of {len(data)}\nFor a percentage of: {(len(correctly_accepted) + len(correctly_not_accepted))/len(data)}')
    print('\n')
    print(f'The number of correct samples: {len(correct)} out of {len(data)} \nFor a percentage of: {len(correct)/len(data)}')

if __name__ == "__main__":
    main()

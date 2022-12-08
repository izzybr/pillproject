# validation
def validate(model, testloader, criterion):

    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    all_lables = []
    all_predictions = []
    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1

            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            # forward pass
            outputs = model(image)
            # calculate the loss
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            # calculate the accuracy
            _, preds = torch.max(outputs.data, 1)
            #print("labels: " + str(labels))
            #print("predictions: " + str(preds))
            all_lables.append(labels.item())
            all_predictions.append(preds.item())
            valid_running_correct += (preds == labels).sum().item()

    # loss and accuracy for the complete epoch
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))

    throttle = 0

    total_mispredicted = 0
    total_correct_predict = 0
    total_images = len(all_lables)

    for idx in all_lables:
        da_label = all_lables[throttle]
        da_prediction = all_predictions[throttle]
        print("label: " + str(da_label))
        print("prediction: " + str(da_prediction))
        if da_label != da_prediction:
            total_mispredicted += 1
        else:
            total_correct_predict += 1
        throttle += 1

    print("TOTAL IMAGES: " + str(total_images))
    print("TOTAL CORRECTLY PREDICTED: " + str(total_correct_predict))
    print("TOTAL INCCORECTLY PREDICTED: " + str(total_mispredicted))

    return epoch_loss, epoch_acc

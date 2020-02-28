database={}

def perform_analytics():
    for form_name,values in database:
        label_list = values["label"]
        checkbox_list = values["checkbox"]
import streamlit as st
import time
import pickle
import pandas as pd
from pbl_prd_py import predict_disease
import warnings

warnings.filterwarnings("ignore")

# Load your data and models (same as in your original code)
data = pd.read_csv('Training.csv')
X = data.drop('prognosis', axis=1)
y = data['prognosis']
des = pd.read_csv('symptom_Description.csv')
pre = pd.read_csv('symptom_precaution.csv')

with open('bayes_dis_model.pkl', 'rb') as bayes:
    bayes_model = pickle.load(bayes)

with open('rf_dis_model.pkl', 'rb') as rf:
    random_forest_model = pickle.load(rf)

with open('xg_dis_model.pkl', 'rb') as xg:
    xgboost_model = pickle.load(xg)
    
    
st.title("MediCuri BOT")
st.subheader('Welcome to MediCuri :smile:', divider='rainbow')

if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "past" not in st.session_state:
    st.session_state.past = []

def add_message(role, content):
    st.session_state.messages.append({"role": role, "content": content})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
              
               
# with st.chat_message('assistant'):
#     st.write('Welcome to MediCuri')

def simulate_typing(text):
    message_placeholder = st.empty()
    full_response = ""
    for chunk in text.split():
        full_response += chunk + " "
        time.sleep(0.05)
        # Add a blinking cursor to simulate typing
        message_placeholder.write(full_response + "▌")
    message_placeholder.write(full_response)

def intro():
    input = st.chat_input('Greeting')
    if input:
        st.session_state.user_greet = input
        add_message("user", input)
        with st.chat_message('user'):
            # st.write(input)
            simulate_typing(input)
        add_message("assistant", 'hey!')
        with st.chat_message('assistant'):
            # st.write(f'hey!')
            simulate_typing('hey!')
        time.sleep(0.1)
        add_message("assistant", 'Tell me the first symptom that you are experiencing the most...')
        with st.chat_message('assistant'):
            # st.write('Tell me the first symptom that you are experiencing the most...')
            simulate_typing('Tell me the first symptom that you are experiencing the most...')
        
def symptom_taking_1():
    sym_input_1 = st.chat_input(f"Symptom_1", key=f'sy_1')
    if sym_input_1 != None:
        st.session_state.past.append(sym_input_1.strip().lower().replace(" ","_"))
    if sym_input_1:
        st.session_state.symp_1 = sym_input_1
        add_message("user", sym_input_1)
        with st.chat_message('user'):
            # st.write(sym_input_1)
            simulate_typing(sym_input_1)
        add_message("assistant", f'Which symptom are you experiencing other than {sym_input_1}')
        with st.chat_message('assistant'):
            # st.write(f'Which symptom are you experiencing other than {sym_input_1}')
            simulate_typing(f'Which symptom are you experiencing other than {sym_input_1}')
            
def symptom_taking_2():
    sym_input_2 = st.chat_input(f"Symptom_2", key=f'sy_2')
    if sym_input_2 != None:
        st.session_state.past.append(sym_input_2.strip().lower().replace(" ","_"))
    if sym_input_2:
        st.session_state.symp_2 = sym_input_2
        add_message("user", sym_input_2)
        with st.chat_message('user'):
            # st.write(sym_input_2)
            simulate_typing(sym_input_2)
        add_message("assistant", f'Which symptom are you experiencing other than {sym_input_2}')
        with st.chat_message('assistant'):
            # st.write(f'Which symptom are you experiencing other than {sym_input_2}')
            simulate_typing(f'Which symptom are you experiencing other than {sym_input_2}')
            
def symptom_taking_3():
    sym_input_3 = st.chat_input(f"Symptom_3", key=f'sy_3')
    if sym_input_3 != None:
        st.session_state.past.append(sym_input_3.strip().lower().replace(" ","_"))
    if sym_input_3:
        st.session_state.symp_3 = sym_input_3
        add_message("user", sym_input_3)
        with st.chat_message('user'):
            # st.write(sym_input_3)
            simulate_typing(sym_input_3)
        add_message("assistant", f'Which symptom are you experiencing other than {sym_input_3}')
        with st.chat_message('assistant'):
            # st.write(f'Which symptom are you experiencing other than {sym_input_3}')
            simulate_typing(f'Which symptom are you experiencing other than {sym_input_3}')
            
def symptom_taking_4():
    sym_input_4 = st.chat_input(f"Symptom_4", key=f'sy_4')
    if sym_input_4 != None:
        st.session_state.past.append(sym_input_4.strip().lower().replace(" ","_"))
    if sym_input_4:
        st.session_state.symp_4 = sym_input_4
        add_message("user", sym_input_4)
        with st.chat_message('user'):
            # st.write(sym_input_4)
            simulate_typing(sym_input_4)
        # add_message("assistant", f'Do you experience any other disease then these?')
        # with st.chat_message('assistant'):
        #     st.write(f'Do you experience any other disease then these?')
            
        
# def option():
#     option_inp = st.chat_input("Yes / No")
#     if option_inp:
#         st.session_state.op = option_inp.lower()
#         add_message("user", option_inp)
#         with st.chat_message('user'):
#             st.write(option_inp)
#         if option_inp == "yes":
#             add_message("assistant", f'Which another symptoms is bothering your health?')
#             with st.chat_message('assistant'):
#                 st.write(f'Which another symptoms is bothering your health?')
#         elif option_inp!= "no":
#             add_message("assistant", "Please Enter valid Input")
    
# def yes():
#     if st.session_state.op == "yes":
#         sym_input_5 = st.chat_input(f"Symptom_5")
#         if sym_input_5 != None:
#             st.session_state.past.append(sym_input_5)
#         if sym_input_5:
#             st.session_state[f'symp_5'] = sym_input_5
#             add_message("user", sym_input_5)
#             with st.chat_message('user'):
#                 st.write(sym_input_5)
#             add_message("assistant", f'Do you experience any other disease then these?')
#             with st.chat_message('assistant'):
#                 st.write(f'Do you experience any other disease then these?')
#     else:
#         pass
            
# def predf():
#     predicted_disease = predict_disease(st.session_state.past)
#     with st.chat_message("assistant"):
#         st.write("Based on the symptoms you provided...")

#         st.write("You may have ", end='')
#         for j in predicted_disease:
#             if predicted_disease[-1] == j:
#                 st.write(j, end='')
#             else:
#                 st.write(f"{j} or ", end='')
#         st.write('\n')
#         st.write("It might not be that bad, but you should take precautions.\n")
#         for j in predicted_disease:
#             description = des[des['Name'] == j].values[0][1]
#             st.write(f"> {j}:\n{description}")
#             precaution = pre[pre['Name'] == j].values
#             st.write(f"Precautions and measures you should take:")
#             for i in range(5):
#                 if i == 0:
#                     continue
#                 else:
#                     st.write(f'{i}) {precaution[0][i].capitalize()}')
#             st.write('\n')

#         st.write("Remember, it's always good to consult with a healthcare professional for a more personalized assessment and tailored advice. Take care!")
#         st.write("Thank you for using the Health Chatbot!")

def predf():
    predicted_disease = predict_disease(st.session_state.past)
    with st.chat_message("assistant"):
        simulate_typing("Based on the symptoms you provided...")

        simulate_typing("You may have ")
        for j in predicted_disease:
            if predicted_disease[-1] == j:
                simulate_typing(j)
            else:
                simulate_typing(f"{j} or ")
        simulate_typing("It might not be that bad, but you should take precautions.\n")
        for j in predicted_disease:
            description = des[des['Name'] == j].values[0][1]
            simulate_typing(f"> {j}:\n{description}")
            precaution = pre[pre['Name'] == j].values
            simulate_typing(f"Precautions and measures you should take:")
            for i in range(5):
                if i == 0:
                    continue
                else:
                    simulate_typing(f'{i}) {precaution[0][i].capitalize()}')
            simulate_typing('\n')

        simulate_typing("Remember, it's always good to consult with a healthcare professional for a more personalized assessment and tailored advice. Take care!")
        simulate_typing("Thank you for using the Health Chatbot!")

if "user_greet" not in st.session_state:  
    intro()
if "user_greet" in st.session_state and "symp_1" not in st.session_state:
    symptom_taking_1()
if "user_greet" in st.session_state and "symp_1" in st.session_state and "symp_2" not in st.session_state:
    symptom_taking_2()
if "user_greet" in st.session_state and "symp_1" in st.session_state and "symp_2" in st.session_state and "symp_3" not in st.session_state:
    symptom_taking_3()
if "user_greet" in st.session_state and "symp_1" in st.session_state and "symp_2" in st.session_state and "symp_3" in st.session_state and "symp_4" not in st.session_state:
    symptom_taking_4()
# if "user_greet" in st.session_state and "symp_1" in st.session_state and "symp_2" in st.session_state and "symp_3" in st.session_state and "symp_4" in st.session_state and "op" not in st.session_state:
#     option()
# if "user_greet" in st.session_state and "symp_1" in st.session_state and "symp_2" in st.session_state and "symp_3" in st.session_state and "symp_4" in st.session_state and "op" in st.session_state and "symp_5" not in st.session_state:
#     yes()
# if "user_greet" in st.session_state and "symp_1" in st.session_state and "symp_2" in st.session_state and "symp_3" in st.session_state and "symp_4" in st.session_state and "op" in st.session_state and "symp_5" in st.session_state:
#     predf()
if "user_greet" in st.session_state and "symp_1" in st.session_state and "symp_2" in st.session_state and "symp_3" in st.session_state and "symp_4" in st.session_state:
    predf()

checkbox_placeholder = st.empty()
agree = checkbox_placeholder.checkbox('Symptoms List')

if agree:
    arr = sorted(X.columns)
    for a in range(len(arr)):
        st.write(f"{a+1}) {arr[a].replace('_',' ').capitalize()}")
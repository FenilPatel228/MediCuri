import logging
import asyncio
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ConversationHandler,
    filters,
    ContextTypes,
)

API_TOKEN=os.environ('API_TOKEN')
import pickle
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from pbl_prd_py import predict_disease



data = pd.read_csv('Final_predict_disease/Training.csv')
X = data.drop('prognosis', axis=1)
y = data['prognosis']
des = pd.read_csv('Final_predict_disease/symptom_Description.csv')
pre = pd.read_csv('Final_predict_disease/symptom_precaution.csv')
sym_list='''
◉ Abdominal pain
◉ Abnormal menstruation
◉ Acidity
◉ Acute liver failure
◉ Altered sensorium
◉ Anxiety
◉ Back pain
◉ Belly pain
◉ Blackheads
◉ Bladder discomfort
◉ Blister
◉ Blood in sputum
◉ Bloody stool
◉ Blurred and distorted vision
◉ Breathlessness
◉ Brittle nails
◉ Bruising
◉ Burning micturition
◉ Chest pain
◉ Chills
◉ Cold hands and feets
◉ Congestion
◉ Constipation
◉ Continuous feel of urine
◉ Continuous sneezing
◉ Cough
◉ Cramps
◉ Dark urine
◉ Dehydration
◉ Depression
◉ Diarrhoea
◉ Dischromic  patches
◉ Distention of abdomen
◉ Dizziness
◉ Drying and tingling lips
◉ Enlarged thyroid
◉ Excessive hunger
◉ Extra marital contacts
◉ Family history
◉ Fast heart rate
◉ Fatigue
◉ Fluid overload
◉ Foul smell of urine
◉ Headache
◉ High fever
◉ Hip joint pain
◉ History of alcohol consumption
◉ Increased appetite
◉ Indigestion
◉ Inflammatory nails
◉ Internal itching
◉ Irregular sugar level
◉ Irritability
◉ Irritation in anus
◉ Itching
◉ Joint pain
◉ Knee pain
◉ Lack of concentration
◉ Lethargy
◉ Loss of appetite
◉ Loss of balance
◉ Loss of smell
◉ Malaise
◉ Mild fever
◉ Mood swings
◉ Movement stiffness
◉ Mucoid sputum
◉ Muscle pain
◉ Muscle wasting
◉ Muscle weakness
◉ Nausea
◉ Neck pain
◉ Nodal skin eruptions
◉ Obesity
◉ Pain behind the eyes
◉ Pain during bowel movements
◉ Pain in anal region
◉ Painful walking
◉ Palpitations
◉ Passage of gases
◉ Patches in throat
◉ Phlegm
◉ Polyuria
◉ Prominent veins on calf
◉ Puffy face and eyes
◉ Pus filled pimples
◉ Receiving blood transfusion
◉ Receiving unsterile injections
◉ Red sore around nose
◉ Red spots over body
◉ Redness of eyes
◉ Restlessness
◉ Runny nose
◉ Rusty sputum
◉ Scurring
◉ Shivering
◉ Silver like dusting
◉ Sinus pressure
◉ Skin peeling
◉ Skin rash
◉ Slurred speech
◉ Small dents in nails
◉ Spinning movements
◉ Spotting  urination
◉ Stiff neck
◉ Stomach bleeding
◉ Stomach pain
◉ Sunken eyes
◉ Sweating
◉ Swelled lymph nodes
◉ Swelling joints
◉ Swelling of stomach
◉ Swollen blood vessels
◉ Swollen extremeties
◉ Swollen legs
◉ Throat irritation
◉ Toxic look (typhos)
◉ Ulcers on tongue
◉ Unsteadiness
◉ Visual disturbances
◉ Vomiting
◉ Watering from eyes
◉ Weakness in limbs
◉ Weakness of one body side
◉ Weight gain
◉ Weight loss
◉ Yellow crust ooze
◉ Yellow urine
◉ Yellowing of eyes
◉ Yellowish skin
'''
with open('Final_predict_disease/bayes_dis_model.pkl', 'rb') as bayes:
    bayes_model = pickle.load(bayes)
    
with open('Final_predict_disease/rf_dis_model.pkl', 'rb') as rf:
    random_forest_model = pickle.load(rf)

with open('Final_predict_disease/xg_dis_model.pkl', 'rb') as xg:
    xgboost_model = pickle.load(xg)

# Initialize logging and other settings
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# Conversation states
SYMPTOM1, SYMPTOM2, SYMPTOM3, SYMPTOM4, SYMPTOMs, PREDICTION = range(6)

# Initialize the list to store symptoms
user_symptoms = []

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Welcome to the Health Chatbot! Please describe the symptoms you are experiencing."
    )
    return SYMPTOM1

async def collect_symptoms_1(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symptom = update.message.text.lower().strip().replace(" ","_")
    
    # Check if the symptom is valid
    if symptom in X.columns:
        if symptom not in user_symptoms:
            user_symptoms.append(symptom)
            # break
        else:
            await update.message.reply_text(f"You've already mentioned {symptom}. Please describe another symptom.")
            return SYMPTOM1
    else:
        await update.message.reply_text(f"Please enter a valid symptom. For reference, take a look at below list...\n{sym_list}")
        return SYMPTOM1

    await update.message.reply_text(f"Please enter a symptom you're experiencing other than {symptom}.")
    return SYMPTOM2

async def collect_symptoms_2(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symptom = update.message.text.lower().strip().replace(" ","_")
        
        # Check if the symptom is valid
    if symptom in X.columns:
        if symptom not in user_symptoms:
            user_symptoms.append(symptom)
        else:
            await update.message.reply_text(f"You've already mentioned {symptom}. Please describe another symptom.")
            return SYMPTOM2
    else:
        await update.message.reply_text(f"Please enter a valid symptom. For reference, take a look at below list...\n{sym_list}")
        return SYMPTOM2
    await update.message.reply_text(f"Please enter a symptom you're experiencing other than {symptom}.")
    return SYMPTOM3

async def collect_symptoms_3(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symptom = update.message.text.lower().strip().replace(" ","_")
    
    # Check if the symptom is valid
    if symptom in X.columns:
        if symptom not in user_symptoms:
            user_symptoms.append(symptom)
        else:
            await update.message.reply_text(f"You've already mentioned {symptom}. Please describe another symptom.")
            return SYMPTOM3
    else:
        await update.message.reply_text(f"Please enter a valid symptom. For reference, take a look at below list...\n{sym_list}")
        return SYMPTOM3
    await update.message.reply_text(f"Please enter a symptom you're experiencing other than {symptom}.")
    return SYMPTOM4

async def collect_symptoms_4(update: Update, context: ContextTypes.DEFAULT_TYPE):
    symptom = update.message.text.lower().strip().replace(" ","_")
    
    # Check if the symptom is valid
    if symptom in X.columns:
        if symptom not in user_symptoms:
            user_symptoms.append(symptom)
        else:
            await update.message.reply_text(f"You've already mentioned {symptom}. Please describe another symptom.")
            return SYMPTOM4
    else:
        await update.message.reply_text(f"Please enter a valid symptom. For reference, take a look at below list...\n{sym_list}")
        return SYMPTOM4
            
    await update.message.reply_text("Are you experiencing any other symptoms? (Yes/No)")
    return SYMPTOMs
    

async def collect_symptoms(update: Update, context: ContextTypes.DEFAULT_TYPE):
    var = update.message.text
    
    if var == "yes" or var == 'Yes':
        await update.message.reply_text("What is the symptom that you're experiencing: ")
        return SYMPTOM4
    elif (var !='yes' and var != 'Yes' and var!='no' and var != 'No'):
        await update.message.reply_text("Invalid input. Please respond with 'Yes' or 'No'.")
        return SYMPTOMs
    await update.message.reply_text("Enter 'Predict' to get prediction.")
    return PREDICTION
        

async def pre_disease(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Predict the disease based on the collected symptoms
    predicted_disease = predict_disease(user_symptoms)

    # Build a response message
    response_message = "Based on the symptoms you provided, you may have: \n"
    response_message += "\n".join(predicted_disease)

    # Provide descriptions and precautions for predicted diseases (you can modify this part)
    for disease in predicted_disease:
        description = des[des['Name'] == disease].values[0][1]
        precautions = pre[pre['Name'] == disease].values[0][1:]

        response_message += f"\n\n{disease}:\n{description}\nPrecautions and measures you should take:\n"
        for i, precaution in enumerate(precautions, start=1):
            response_message += f"{i}. {precaution}\n"

    # Send the response message to the user
    await update.message.reply_text(response_message)

    # Clear the user's symptoms for the next conversation
    user_symptoms.clear()

    # End the conversation
    return ConversationHandler.END

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('Conversation canceled. Type /start to begin a new conversation.')
    return ConversationHandler.END

def main():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    application = Application.builder().token(API_TOKEN).build()

    conversation_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            SYMPTOM1: [MessageHandler(filters.TEXT & ~filters.COMMAND, collect_symptoms_1)],
            SYMPTOM2: [MessageHandler(filters.TEXT & ~filters.COMMAND, collect_symptoms_2)],
            SYMPTOM3: [MessageHandler(filters.TEXT & ~filters.COMMAND, collect_symptoms_3)],
            SYMPTOM4: [MessageHandler(filters.TEXT & ~filters.COMMAND, collect_symptoms_4)],
            SYMPTOMs: [MessageHandler(filters.TEXT & ~filters.COMMAND, collect_symptoms)],
            PREDICTION: [MessageHandler(filters.TEXT & ~filters.COMMAND, pre_disease)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )

    application.add_handler(conversation_handler)

    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()




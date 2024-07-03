import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime
import pyttsx3



def send_mail(toaddr, attachments, license_no, timestamp):
    fromaddr = "mescoe1@outlook.com"
    pswd = "dummy@987"
    # toaddr = "yerkalsm19.comp@coep.ac.in"

    dt = datetime.fromtimestamp(timestamp / 1000)
    time_str = dt.strftime("%d-%m-%Y %H:%M:%S")
    
    # instance of MIMEMultipart
    msg = MIMEMultipart()

    # storing the senders email address
    msg['From'] = fromaddr

    # storing the receivers email address
    msg['To'] = toaddr

    # storing the subject
    msg['Subject'] = "Your vehicle detected during surveillance !!!"

    # string to store the body of the mail
    body = "This email is sent by MESCOE security surveillance team. \nYour vehicle with LICENSE NUMBER : " + license_no + " is detected at TIME: " + time_str + "\nWe have also attached the corresponding picture of your vehicle and its License Plate. \n\nThanks and Regards,\nMESCOE Security Team."

    # attach the body with the msg instance
    msg.attach(MIMEText(body, 'plain'))

    frame_img = MIMEBase('application', 'octet-stream')
    filename = "Vehicle Image.jpg"
    frame_img.set_payload(attachments[0].tobytes())
    encoders.encode_base64(frame_img)
    frame_img.add_header('Content-Disposition', "attachment; filename= %s" % filename)
    msg.attach(frame_img)

    license_plate_img = MIMEBase('application', 'octet-stream')
    filename = "License Plate.jpg"
    license_plate_img.set_payload(attachments[1].tobytes())
    encoders.encode_base64(license_plate_img)
    license_plate_img.add_header('Content-Disposition', "attachment; filename= %s" % filename)
    msg.attach(license_plate_img)

    # creates SMTP session
    s = smtplib.SMTP('smtp.office365.com',587)

    # start TLS for security
    s.starttls()

    # Authentication
    s.login(fromaddr, pswd)

    # Converts the Multipart msg into a string
    text = msg.as_string()

    # sending the mail
    s.sendmail(fromaddr, toaddr, text)

    # terminating the session
    s.quit()

def generate_voice(lic_no):
    # Create a new text-to-speech engine
    engine = pyttsx3.init()

    # Set the voice
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id) # Change the index number to select a different voice

    # Set the speed (optional)
    engine.setProperty('rate', 150) # Change the value to adjust the speed

    # Enter the text to be spoken
    text = "License Number " + lic_no + " is detected."
    engine.runAndWait()

    engine.say(text)
    engine.runAndWait()

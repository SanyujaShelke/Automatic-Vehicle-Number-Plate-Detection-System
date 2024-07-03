from tkinter import *
from tkinter import ttk
from ttkthemes import themed_tk as tk
import tkinter.font as font
from tkinter import messagebox
from functools import partial
from database import *
import PIL.Image, PIL.ImageTk
from tkinter import filedialog as fd
import threading
import time
from surveillance import start_video_surveillance


window = Tk()
# window = tk.ThemedTk()
# window.get_themes()
# window.set_theme("radiance")
window.title("License Plate Detector")
window.geometry('800x800')
window.configure(bg='white')
logo = PIL.ImageTk.PhotoImage(PIL.Image.open('images\\Style1\\logo.png'))
window.iconphoto(False, logo)
index = 0
thread_flag = 1 

#loading images
# img1 = PIL.ImageTk.PhotoImage(PIL.Image.open('images\\Login-page2.png'))

# img2 = PIL.ImageTk.PhotoImage(PIL.Image.open('images\\Login-button.png'))
# img1 = PIL.ImageTk.PhotoImage(PIL.Image.open('images\\Style1\\ProjectLogo1.jpg'))
img1 = PIL.ImageTk.PhotoImage(PIL.Image.open('images\\Style1\\HomeLogo12.png'))
login1 = PIL.ImageTk.PhotoImage(PIL.Image.open('images\\Style1\\button_login.png'))
img2 = PIL.ImageTk.PhotoImage(PIL.Image.open('images\\Login-button.png'))
admin_logo = PIL.ImageTk.PhotoImage(PIL.Image.open('images\\Style1\\admin1.jpg'))
student_login1 = PIL.ImageTk.PhotoImage(PIL.Image.open('images\\Style1\\student_login.jpg'))
addQuestionPaper_logo = PIL.ImageTk.PhotoImage(PIL.Image.open('images\\Style1\\DETECTION.png'))
addNewUser_logo = PIL.ImageTk.PhotoImage(PIL.Image.open('images\\Style1\\add-user.png'))
save1 = PIL.ImageTk.PhotoImage(PIL.Image.open('images\\Style1\\button_save.png'))
save2 = PIL.ImageTk.PhotoImage(PIL.Image.open('images\\Style1\\button_save1.png'))
save3 = PIL.ImageTk.PhotoImage(PIL.Image.open('images\\Style1\\button_save3.png'))
submit1 = PIL.ImageTk.PhotoImage(PIL.Image.open('images\\Style1\\button_submit.png'))
submit2 = PIL.ImageTk.PhotoImage(PIL.Image.open('images\\Style1\\button_submit1.png'))
generatePaper1 = PIL.ImageTk.PhotoImage(PIL.Image.open('images\\Style1\\button_generate-paper.png'))
quit1 = PIL.ImageTk.PhotoImage(PIL.Image.open('images\\Style1\\button_quit.png'))
quit2 = PIL.ImageTk.PhotoImage(PIL.Image.open('images\\Style1\\button_quit1.png'))
start_paper1 = PIL.ImageTk.PhotoImage(PIL.Image.open('images\\Style1\\rv2.png'))
change_password = PIL.ImageTk.PhotoImage(PIL.Image.open('images\\Style1\\cp2.png'))
change_password1 = PIL.ImageTk.PhotoImage(PIL.Image.open('images\\Style1\\button_change-password1.png'))
cancel = PIL.ImageTk.PhotoImage(PIL.Image.open('images\\Style1\\button_cancel.png'))
next1 = PIL.ImageTk.PhotoImage(PIL.Image.open('images\\Style1\\button_next.png'))
prev1 = PIL.ImageTk.PhotoImage(PIL.Image.open('images\\Style1\\button_prev.png'))
endtest1 = PIL.ImageTk.PhotoImage(PIL.Image.open('images\\Style1\\button_end-test.png'))
adduser = PIL.ImageTk.PhotoImage(PIL.Image.open('images\\Style1\\button_add-user2.png'))

def buttonclick_adduser():
    if main_frame:
        main_frame.pack_forget()
    if new_frame:
            new_frame.pack_forget()
    global postlogin_frame_a
    if postlogin_frame_a:
        postlogin_frame_a.pack_forget()
    
    postlogin_frame_a = LabelFrame(window, text = "add user", padx =10, pady =10,bg="white",border=0,borderwidth=0)
    postlogin_frame_a.pack(padx = 10, pady = 4)
    
    Label(postlogin_frame_a ,bg="white",text = "Role", padx=10,pady=5).grid(row = 0,column = 0)
    Label(postlogin_frame_a ,bg="white",text = "Username", padx=10,pady=5).grid(row = 1,column = 0)
    Label(postlogin_frame_a ,bg="white",text = "Password", padx=10,pady=5).grid(row = 2,column = 0)

    newrole_var = StringVar() 
    newrole = ttk.Combobox(postlogin_frame_a, width = 25, textvariable = newrole_var) 
    newrole['values'] = ('Admin', 'Student', 'Staff', 'Worker') 
    newrole.grid(row = 0,column = 1)

    newusername_var = StringVar()
    newusername = Entry(postlogin_frame_a, width = 28, textvariable = newusername_var)
    newusername.grid(row = 1,column = 1)

    newpassword_var = StringVar()
    newpassword = Entry(postlogin_frame_a, show ='*', width = 28, textvariable = newpassword_var)
    newpassword.grid(row = 2,column = 1)

    newlogin_btn = Button(postlogin_frame_a ,border=0,borderwidth=0,image=adduser, command = lambda: add_user(newusername_var.get(), newpassword_var.get(), newrole_var.get()))
    newlogin_btn.grid(row=3,column=0)

    quit_btn = Button(postlogin_frame_a, image=quit2,border=0,borderwidth=0, command = window.destroy)
    quit_btn.grid(row=3,column=1)


def buttonclick_changepassword(username, change_password_frame):

    change_password_frame.pack(padx = 10, pady = 4)

    Label(change_password_frame ,bg="white",font = ('arial', '20', 'normal'),text = "Enter Old password", padx=10,pady=5).grid(row = 0,column = 0)
    Label(change_password_frame ,bg="white",font = ('arial', '20', 'normal'),text = "Enter New password", padx=10,pady=5).grid(row = 1,column = 0)
    Label(change_password_frame ,bg="white",font = ('arial', '20', 'normal'),text = "Confirm Password", padx=10,pady=5).grid(row = 2,column = 0)

    password1 = Entry(change_password_frame,font = ('arial', '20', 'normal'), show ='*', width = 28)
    password1.grid(row = 0,column = 1)

    password2 = Entry(change_password_frame,font = ('arial', '20', 'normal'), show ='*', width = 28)
    password2.grid(row = 1,column = 1)

    password3 = Entry(change_password_frame,font = ('arial', '20', 'normal'), show ='*', width = 28)
    password3.grid(row = 2,column = 1)

    def buttonclick_updatepassword():
        if not check_user(username, password1.get(), 'Student'):
            reponse = messagebox.showerror("Error", "Incorrect password")

        elif password2.get() != password3.get():
            reponse = messagebox.showerror("Error", "Confirm password is not correct")
        else:
            update_password(username, password2.get(), 'Student')
            messagebox.showinfo("showinfo", "Data entry added succesfully")
            change_password_frame.destroy()
            pass 
            # query = '''UPDATE login
            #             SET password = ?
            #             where username = ?'''
            # cursor.execute(query, (password2.get(), username))
            # conn.commit()


    changepassword_btn = Button(change_password_frame, image=change_password1,bg="white",border=0,borderwidth=0, command = buttonclick_updatepassword)
    changepassword_btn.grid(row=5,column=0, padx =10, pady =10)

    
    changepassword_btn = Button(change_password_frame,image=cancel,bg="white",border=0,borderwidth=0, command = change_password_frame.destroy)
    changepassword_btn.grid(row=5,column=1, padx =10, pady =10)


def buttonclick_add_details(username, role, change_password_frame):

    if check_if_user_details_exist(username):
        messagebox.showerror("Error", "Details already exist")
        return

    change_password_frame.pack(padx = 10, pady = 4)

    Label(change_password_frame ,bg="white",font = ('arial', '10', 'normal'),text = "Enter vehicle number", padx=10,pady=5).grid(row = 0,column = 0)
    Label(change_password_frame ,bg="white",font = ('arial', '10', 'normal'),text = "Enter your mobile number", padx=10,pady=5).grid(row = 1,column = 0)
    Label(change_password_frame ,bg="white",font = ('arial', '10', 'normal'),text = "Enter your email", padx=10,pady=5).grid(row = 2,column = 0)
    Label(change_password_frame ,bg="white",font = ('arial', '10', 'normal'),text = "Enter college name", padx=10,pady=5).grid(row = 3,column = 0)
    Label(change_password_frame ,bg="white",font = ('arial', '10', 'normal'),text = "Enter your department", padx=10,pady=5).grid(row = 4,column = 0)
    
    if role in ('Student', 'Staff') :
        Label(change_password_frame ,bg="white",font = ('arial', '10', 'normal'),text = "Enter your class", padx=10,pady=5).grid(row = 5,column = 0)

    data1 = Entry(change_password_frame,font = ('arial', '10', 'normal'),width = 28)
    data1.grid(row = 0,column = 1)

    data2 = Entry(change_password_frame,font = ('arial', '10', 'normal'), width = 28)
    data2.grid(row = 1,column = 1) 
    
    data3 = Entry(change_password_frame,font = ('arial', '10', 'normal'), width = 28)
    data3.grid(row = 2,column = 1)
    
    data4 = Entry(change_password_frame,font = ('arial', '10', 'normal'),width = 28)
    data4.grid(row = 3,column = 1)

    data5 = Entry(change_password_frame,font = ('arial', '10', 'normal'), width = 28)
    data5.grid(row = 4,column = 1)

    if role in ('Student', 'Staff') :
        data6 = Entry(change_password_frame,font = ('arial', '10', 'normal'), width = 28)
        data6.grid(row = 5,column = 1)

    def buttonclick_add_details():
        
        if role in ('Student', 'Staff') :
            cls = data6.get()
        else:
            cls = ''
        add_user_details(username, data1.get(), data2.get(), data3.get(), data4.get(), data5.get(), cls)
        messagebox.showinfo("showinfo", "Data entry added succesfully")
        change_password_frame.destroy()

    changepassword_btn = Button(change_password_frame, image=save3,bg="white",border=0,borderwidth=0, command = buttonclick_add_details)
    changepassword_btn.grid(row=6,column=0, padx =10, pady =10)

    
    changepassword_btn = Button(change_password_frame,image=cancel,bg="white",border=0,borderwidth=0, command = change_password_frame.destroy)
    changepassword_btn.grid(row=6,column=1, padx =10, pady =10)


  
def buttonclick_start_detection():
    
    details_frame = LabelFrame(window, padx =10, pady =10)
    details_frame.pack(padx = 10, pady = 4)

    entry_list = []

    e = Entry(details_frame, width=15, fg='black', font=('Arial',10,'bold'))
    e.grid(row=0, column=0)
    e.insert(END, "Username")
    
    e = Entry(details_frame, width=15, fg='black', font=('Arial',10,'bold'))
    e.grid(row=0, column=1)
    e.insert(END, "Name plate")
    
    e = Entry(details_frame, width=30, fg='black', font=('Arial',10,'bold'))
    e.grid(row=0, column=2)
    e.insert(END, "timestamp")

    e = Entry(details_frame, width=15, fg='black', font=('Arial',10,'bold'))
    e.grid(row=0, column=3)
    e.insert(END, "College")
    
    e = Entry(details_frame, width=15, fg='black', font=('Arial',10,'bold'))
    e.grid(row=0, column=4)
    e.insert(END, "Department")
    
    e = Entry(details_frame, width=15, fg='black', font=('Arial',10,'bold'))
    e.grid(row=0, column=5)
    e.insert(END, "class")

    for i in range(5):
        curr_list = []
        for j in range(6):
            if j != 2:
                e = Entry(details_frame, width=15, fg='black',
                            font=('Arial',10))
            else:
                e = Entry(details_frame, width=30, fg='black',
                            font=('Arial',10))

            e.grid(row=i+1, column=j)
            curr_list.append(e)
        entry_list.append(curr_list)
    # print(entry_list)

    def update_data():
        while(thread_flag):
            data = get_last_5_records()
            # print(data)
            for i in range(min(5, len(data))):          

                for j in range(6):
                    # print(3*i + j)
                    if j!=2:
                        entry_list[i][j].delete(0, END)
                        entry_list[i][j].insert(END, data[i][j])
                    else:
                        entry_list[i][j].delete(0, END)
                        entry_list[i][j].insert(END, time.ctime(data[i][j] / 1000))

            time.sleep(0.5)
            
    t1 = threading.Thread(target=update_data)
    t1.start()
    
    t2 = threading.Thread(target=start_video_surveillance, args=(r'./test_dataset/trim.mp4',))
    t2.start()    
    

postlogin_frame_a = 0
def buttonclick_login():
   
    if not check_user(username.get(), password.get(), role.get()):
        reponse = messagebox.showerror("Error", "Incorrect username or password")
        return
    login_frame.pack_forget()

    welcome_frame = LabelFrame(window, padx =10, pady =10)
    welcome_frame.pack(padx = 10, pady = 4)
    # Label(welcome_frame ,text = "Welcome " + username.get(), padx=10,pady=5).grid(row = 0,column = 0)

    if role.get() == 'Admin':
        welcome_frame = LabelFrame(window, padx =10,bg="white",borderwidth=0, border=0)
        welcome_frame.pack(padx = 10, pady = 4)
        Label(welcome_frame , image = admin_logo, padx=5,pady=5,borderwidth = 0,border=0).grid(row = 0,column = 0, columnspan = 2, pady = 20)
        Label(welcome_frame ,text = "Welcome Admin" , padx=10,pady=5,bg="white",font = ('arial', '15', 'normal')).grid(row = 1,column = 1)
        option_frame = LabelFrame(window, padx =20, pady =20,bg="white",borderwidth=0, border=0)
        option_frame.pack(padx = 10, pady = 4)

        qpaper_btn = Button(option_frame ,image=addQuestionPaper_logo,border =0,borderwidth =0,command = buttonclick_start_detection)
        qpaper_btn.grid(row=0,column=0,padx=20)

        adduser_btn = Button(option_frame ,image=addNewUser_logo,border =0,borderwidth =0,command = buttonclick_adduser)
        adduser_btn.grid(row=0,column=3)
    elif role.get() in ('Student', 'Staff', 'Worker') :
        welcome_frame = LabelFrame(window, padx =10,bg="white",borderwidth=0, border=0)
        welcome_frame.pack(padx = 10, pady = 4)
        Label(welcome_frame , image = student_login1, padx=5,pady=5,borderwidth = 0,border=0).grid(row = 0,column = 0, columnspan = 2, pady = 20)
        Label(welcome_frame ,text = "Welcome Student" , padx=10,pady=5,bg="white",font = ('arial', '10', 'normal')).grid(row = 1,column = 1)
        option_frame = LabelFrame(window, padx =10, pady =10,bg="white",border=0,borderwidth=0)
        option_frame.pack(padx = 10, pady = 4)

        change_password_frame = LabelFrame(window, padx =10, pady =10,bg="white",border=0,borderwidth=0)
        
        qpaper_btn = Button(option_frame ,image=start_paper1,border=0,borderwidth=0, command = partial(buttonclick_add_details, username.get(), role.get(), change_password_frame))
        qpaper_btn.grid(row=0,column=0)

        changepassword_btn = Button(option_frame ,image=change_password,border=0,borderwidth=0, command = partial(buttonclick_changepassword, username.get(), change_password_frame))
        changepassword_btn.grid(row=0,column=1, padx =10, pady =10)
        pass

def new_frame_destroy(new_frame, save_btn):
    save_btn["state"] = NORMAL
    new_frame.destroy()


main_frame = 0
new_frame = 0

login_frame = LabelFrame(window, padx =10, pady =10,bg ="white",borderwidth=0,border=0)
login_frame.pack(padx = 10, pady = 4)


Label(login_frame , image = img1, padx=10,pady=15).grid(row = 0,column = 0, columnspan = 2, pady = 20)
Label(login_frame ,text = "Role", padx=10,pady=5,bg="white", font = ('arial', '20', 'normal')).grid(row = 1,column = 0)
Label(login_frame ,text = "Username", padx=10,pady=5,bg="white", font = ('arial', '20', 'normal')).grid(row = 2,column = 0)
Label(login_frame ,text = "Password", padx=10,pady=5,bg="white", font = ('arial', '20', 'normal')).grid(row = 3,column = 0)

role_var = StringVar() 
role = ttk.Combobox(login_frame,height =30,width = 25,font = ('arial', '20', 'normal'),textvariable = role_var) 
role['values'] = ('Admin', 'Student', 'Staff', 'Worker') 
role.grid(row = 1,column = 1)

username_var = StringVar()
username = Entry(login_frame, width = 28, font = ('arial', '20', 'normal'),textvariable = username_var)
username.grid(row = 2,column = 1)

password_var = StringVar()
password = Entry(login_frame, width = 28, font = ('arial', '20', 'normal'),show = '*', textvariable = password_var)
password.grid(row = 3,column = 1)

login_btn = Button(login_frame ,image = login1, border=0,borderwidth=0, command = buttonclick_login)
login_btn.grid(row=4,column=0, columnspan = 2, pady =20)

window.mainloop()

thread_flag = 0

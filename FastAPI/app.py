from fastapi import FastAPI
from model import NewStudent, UpdateStudent
app = FastAPI(
    title="Test app",
    description="FastAPI tutorial"
)

Student ={
    1: {
        "Name":"Arjun",
        "age":"23"
    },
    2:{
        "Name":"Rahul",
        "age":"22"
    }
}

@app.get("/")
def welcome():
    return "Welcome to fastapi"

@app.get("/all-students")
def get_all_students():
    return Student

@app.get("/student/{stud_id}")
def get_sudent_by_id(stud_id: int):
    if stud_id not in Student:
        return f"No student found with student id: {stud_id}"
    else:
        return Student[stud_id]
    
@app.post("/student/register")
def register(stud:NewStudent):
    if not Student:
        new_id = 1
    else:
        new_id = max(Student.keys()) + 1
    Student[new_id] = stud.model_dump()
    return Student[new_id]
    
@app.put("/student/{stud_id}")
def update_student(stud_id: int, stud:UpdateStudent):
    if stud_id not in Student:
        return f"No student found with student id: {stud_id}"
    if stud.name is not None:
        Student[stud_id]["Name"] = stud.name
    if stud.age is not None:
        Student[stud_id]["Age"] = stud.age

@app.delete("/delete-student/{stud_id}")
def delete_student(stud_id:int):
    if stud_id not in Student:
        return f"No student found with student id: {stud_id}"
    del Student[stud_id]
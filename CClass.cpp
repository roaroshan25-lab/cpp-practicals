
#include <iostream>
using namespace std;

class Student
{
private:
    int rollNo;
    string name;

public:
    // Constructor
    Student(int r, string n)
    {
        rollNo = r;
        name = n;
        cout << "Student object created" << endl;
    }

    // Member function
    void display()
    {
        cout << "Roll Number: " << rollNo << endl;
        cout << "Name: " << name << endl;
    }

    // Destructor
    ~Student()
    {
        cout << "Student object destroyed" << endl;
    }
};

int main()
{
    int r;
    string n;

    cout << "Enter roll number: ";
    cin >> r;

    cout << "Enter name: ";
    cin >> n;

    Student s1(r, n);

    s1.display();

    return 0;
}

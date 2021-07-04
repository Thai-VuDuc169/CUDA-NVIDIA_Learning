#include <iostream>
#include <cassert>
using namespace std;

class A 
{
   public:
      int *x;
   public:
      A()
      {
         this->x = new int(5);
         cout << &(this->x) << " : " << this->x << " : " << *(this->x) << endl;  
      }
      ~A()
      {
         delete this->x;
      }

};

int* x()
{
   // int *x = (int*)malloc(sizeof(int));
   int *x = new int;
   return x;
};

int main()
{
   // int *y = x();
   // *y = 1;
   // cout << &y << " : " << y << " : " << *y << endl;

   // delete (y);
   // // assert( *y != NULL);
   // cout << &y << " : " << y << " : " << *y << endl;

   A* a = new A;
   cout << &(a->x) << " : " << a->x << " : " << *(a->x) << endl;
   cout << sizeof(float)  ;
   delete a;

   return 0;
}
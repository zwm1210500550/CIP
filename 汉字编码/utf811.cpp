#include<iostream>
#include<fstream>
using namespace std;

int main()
{

	/*char data[100];
	ofstream f1;
	f1.open("text.txt");
	cout << "please write to the file" << endl;
	cin.getline(data, 100);
	f1 << data << endl;
	f1.close();*/
	ofstream f1;
	f1.open("result.txt");
	char ch;
	char arr[5];
    fstream f2;
	f2.open("text.txt");
	ch = f2.get();
	//while (ch !=EOF)
	while (!f2.eof())
	{
		int temp = ch & 0xf0;
		//cout << temp << endl;
		if (temp < 0xc0)//占一个字节
		{
			//cout << ch << " ";
			f1 << ch<<" ";
			ch = f2.get();
		}
		else if (temp >= 0xc0 && temp < 0xe0)//占两个字节
		{
			arr[0] = ch;
			arr[1] = f2.get();
			arr[2] = '\0';
			f1 << arr<<" ";
			memset(arr, 0, sizeof(arr));
			ch = f2.get();
			/*cout << ch;
			ch = f2.get();
			cout << ch;
			ch = f2.get();
			cout << "2";*/
		}
		else if (temp >= 0xe0 && temp < 0xf0)//占三个字节
		{
			/*cout << ch;
			ch = f2.get();
			cout << ch;
			ch = f2.get();
			cout << ch;
			ch = f2.get();
			cout << "3";*/
			arr[0] = ch;
			arr[1] = f2.get();
			arr[2] = f2.get();
			arr[3] = '\0';
			f1 << arr << " ";
			memset(arr, 0, sizeof(arr));
			ch = f2.get();

		}
		else//占四个字节
		{
			
			arr[0] = ch;
			arr[1] = f2.get();
			arr[2] = f2.get();
			arr[3] = f2.get();
			arr[4] = '\0';
			//cout << arr[0]<<arr[1]<<arr[2]<<arr[3]<<"4";
			f1 << arr << " ";
			memset(arr, 0, sizeof(arr));
			ch = f2.get();
		}
	}
	f1.close();
	f2.close();
	return 0;

}

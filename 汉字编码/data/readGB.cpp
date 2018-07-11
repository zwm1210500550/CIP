#include<stdio.h>
int main(void){
	FILE *fp,*fp2;
	char buffer[3];
	char ch;
	int count=0;
	char* filename="demo2.txt";
	if((fp=fopen(filename,"r+"))==NULL){
		printf("打开文件错误\n");
		fclose(fp);
		return 0; 
	}
	if((fp2=fopen("output2.txt","w+"))==NULL){//输出文件 
		printf("打开文件错误\n");
		fclose(fp2);
		return 0; 
	}
	ch=fgetc(fp);
	while(ch!=EOF){
		buffer[0]=ch;
		fputc(ch,fp2);
		if(ch>127||ch<0){//字符不在ASCII码范围内，读取两个Byte的汉字 
			ch=fgetc(fp);
			buffer[1]=ch;
			fputc(ch,fp2);
			buffer[2]='\0';
			printf("%s ",buffer);
		}
		else
			printf("%c ",buffer[0]);//字符在ASCII码范围内 ，读取一个Byte的字符 
		fputc(' ',fp2);	
		count++;
		ch=fgetc(fp);
	}
	printf("%d\n",count); 
	fprintf(fp2,"%d",count);
	fclose(fp);
	return 1;
} 
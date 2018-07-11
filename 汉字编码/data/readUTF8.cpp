#include<stdio.h>
int getByteType(int ch){//判断UTF-8编码的长度（字节数） 
	if((ch>>7)==0)return 1;
	else if(((ch>>5)&1)==0)return 2;
	else if(((ch>>4)&1)==0)return 3;
	else if(((ch>>3)&1)==0)return 4;
	else return 0;
} 
int main(void){
	FILE *fp=NULL,*fp2=NULL;;
	char* filename="demo.txt";
	char buffer[5];
	char ch;
	int count=0;//汉字等字符总数 
	if((fp=fopen(filename,"r"))==NULL){
		printf("打开文件错误\n");
		fclose(fp);
		return 0;
	}
	if((fp2=fopen("output.txt","w+"))==NULL){
		printf("打开文件错误\n");
		fclose(fp2);
		return 0;
	}
	fread(buffer,sizeof(char),3,fp);
	if(buffer[0]==0xEF && buffer[1]== 0xBB && buffer[2]==0xBF);//是否跳过文件头BOM 
	else fseek(fp,0,SEEK_SET);
	ch=fgetc(fp);
	while(ch!=EOF){		
		int n=getByteType(ch);
		if(n>0){
			count++;
			while((--n)>=0){
				fputc(ch,fp2);		
				ch=fgetc(fp);
				if(n==0){
					fputc(' ',fp2);
				}
			}
		}
	}
	fprintf(fp2,"%d",count);
	printf("%d",count);
	fclose(fp);
	fclose(fp2);
	return 0;
}

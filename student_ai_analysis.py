import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. الإعدادات العامة (Configuration)
# ==========================================
# إنشاء مجلد لحفظ الرسوم البيانية إذا لم يكن موجوداً
OUTPUT_DIR = 'portfolio_visualizations'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# توحيد النمط البصري لجميع الرسوم باستخدام Seaborn
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (9, 5)
plt.rcParams['font.size'] = 11

# ==========================================
# 2. دوال التحليل والرسم (Functions)
# ==========================================

def load_and_prep_data(filepath):
    """قراءة البيانات والتأكد من جاهزيتها"""
    try:
        df = pd.read_csv(filepath)
        print(f"✅ تم تحميل البيانات بنجاح: {df.shape[0]} صف، {df.shape[1]} عمود.")
        return df
    except FileNotFoundError:
        print(f"❌ خطأ: الملف {filepath} غير موجود.")
        return None

def plot_gpa_impact(df):
    """رسم تأثير الذكاء الاصطناعي على المعدل التراكمي حسب التخصص"""
    gpa_by_major = df[['Pre_Semester_GPA','Post_Semester_GPA', 'Major_Category']].groupby('Major_Category').mean().round(2)
    
    ax = gpa_by_major.plot(kind='bar', color=["#8ca43c","#2E350B"], edgecolor='black')
    
    plt.title('تأثير الذكاء الاصطناعي على المعدل', fontweight='bold', pad=15)
    plt.xlabel('التخصصات')
    plt.ylabel('المعدل')
    plt.ylim(2.5, 4.0) # تحسين نطاق الرؤية
    plt.xticks(rotation=0)
    plt.legend(['المعدل السابق', 'المعدل الحالي'])
    plt.tight_layout()
    
    plt.savefig(f'{OUTPUT_DIR}/1_gpa_impact.png', dpi=300)
    plt.close()
    gpa_by_major.to_csv('portfolio_visualizations/summary_gpa_by_major.csv', index=False )
    print("- تم حفظ رسم تأثير المعدل.")

def plot_burnout_risk(df):
    """رسم علاقة الاحتراق النفسي بساعات الدراسة ومستويات القلق"""
    # ترتيب المستويات منطقياً للحصول على قراءة صحيحة للبيانات
    burnout_df = df[['Burnout_Risk_Level','Traditional_Study_Hours','Weekly_GenAI_Hours','Anxiety_Level_During_Exams']]\
                 .groupby('Burnout_Risk_Level').mean()\
                 .reindex(['Low', 'Medium', 'High']).round(2)

    burnout_df.plot(kind='bar', color=["#f97d7d","#ff2d3f","#c00010"], edgecolor='black')
    
    plt.title('مؤشرات الاحتراق النفسي والقلق', fontweight='bold', pad=15)
    plt.xlabel('مستوى خطر الاحتراق')
    plt.ylabel('متوسط القيمة')
    plt.xticks(rotation=0)
    plt.tight_layout()
    
    plt.savefig(f'{OUTPUT_DIR}/2_burnout_risk.png', dpi=300)
    plt.close()
    burnout_df.to_csv('portfolio_visualizations/summery_burnout_df.csv', index=False)
    print("- تم حفظ رسم الاحتراق النفسي.")

def plot_prompt_skills(df):
    """رسم توزيع مهارات هندسة الأوامر حسب التخصص"""
    prompt_skill = pd.crosstab(df['Major_Category'], df['Prompt_Engineering_Skill'])
    
    prompt_skill.plot(kind='bar', edgecolor='black', cmap='Blues')
    
    plt.title('مهارة Prompt Engineering حسب التخصص', fontweight='bold', pad=15)
    plt.xlabel('التخصص')
    plt.ylabel('عدد الطلاب')
    plt.xticks(rotation=0)
    plt.tight_layout()
    
    plt.savefig(f'{OUTPUT_DIR}/3_prompt_skills.png', dpi=300)
    plt.close()
    prompt_skill.to_csv('portfolio_visualizations/summery_prompt_skill.csv', index=False)
    print("- تم حفظ رسم مهارات الأوامر.")

def plot_study_hours_comparison(df):
    """مقارنة ساعات الدراسة التقليدية بالذكاء الاصطناعي"""
    study_hours = df[['Traditional_Study_Hours','Weekly_GenAI_Hours','Major_Category']].groupby('Major_Category').mean().round(2)
    
    study_hours.plot(kind='bar', color=['#8e44ad','#2980b9'], edgecolor='black')
    
    plt.title('مقارنة ساعات الدراسة التقليدية والذكاء الاصطناعي', fontweight='bold', pad=15)
    plt.xlabel('التخصص الأكاديمي')
    plt.ylabel('متوسط الساعات الأسبوعية')
    plt.xticks(rotation=0)
    plt.legend(['الدراسة التقليدية', 'ساعات الذكاء الاصطناعي'])
    plt.tight_layout()
    
    plt.savefig(f'{OUTPUT_DIR}/4_study_hours.png', dpi=300)
    plt.close()
    study_hours.to_csv('portfolio_visualizations/summery_study_hours.csv' , index=False)
    print("- تم حفظ رسم مقارنة الساعات.")

def plot_correlations(df):
    """رسم خريطة الارتباط للمتغيرات الرقمية"""
    corr = df.select_dtypes(include='number').corr()
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('خريطة ارتباط المتغيرات (Correlation Heatmap)', fontweight='bold', pad=15)
    plt.tight_layout()
    
    plt.savefig(f'{OUTPUT_DIR}/5_correlation_heatmap.png', dpi=300)
    plt.close()
    print("- تم حفظ خريطة الارتباط.")

def plot_scatter_relations(df):
    """رسم العلاقات المباشرة (Scatter Plots)"""
    # الرسم الأول: الذكاء الاصطناعي والمعدل
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x='Weekly_GenAI_Hours', y='Post_Semester_GPA', alpha=0.5, color='#2980b9')
    plt.title('ساعات الذكاء الاصطناعي مقابل المعدل الحالي', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/6a_scatter_ai_gpa.png', dpi=300)
    plt.close()

    # الرسم الثاني: الذكاء الاصطناعي والدراسة التقليدية
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x='Weekly_GenAI_Hours', y='Traditional_Study_Hours', alpha=0.5, color='#8e44ad')
    plt.title('ساعات الذكاء الاصطناعي مقابل ساعات الدراسة التقليدية', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/6b_scatter_ai_traditional.png', dpi=300)
    plt.close()
    print("- تم حفظ رسوم العلاقات (Scatter).")

# ==========================================
# 3. التشغيل الرئيسي (Main Execution Block)
# ==========================================
if __name__ == "__main__":
    file_path = 'ai_student_impact_dataset (1).csv'
    
    # تنفيذ الكود بخطوات متسلسلة
    print("بدء تحليل البيانات واستخراج الرسوم...")
    df = load_and_prep_data(file_path)
    
    if df is not None:
        plot_gpa_impact(df)
        plot_burnout_risk(df)
        plot_prompt_skills(df)
        plot_study_hours_comparison(df)
        plot_correlations(df)
        plot_scatter_relations(df)
        print(f"\n🎉 تمت العملية بنجاح! تم حفظ جميع الرسوم بجودة عالية في مجلد: '{OUTPUT_DIR}'")

        




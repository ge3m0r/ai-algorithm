# 练习1：用户画像的模拟与生成• 目标：通过编程自动生成一批“用户画像”数据，掌握数据合成与基本属性建模方法。• 应用场景：推荐系统、用户分析、机器学习建模等。
# • 任务要求：
# • 明确什么是“用户画像”及其关键属性。
# • 定义属性类型与取值范围。
# • 编写脚本，批量生成模拟用户数据。
# 属性名 类型 说明与取值示例性别 类别型 男、女、未透露所在城市 类别型 北京、上海、广州、深圳、其他消费水平 类别型 高、中、低年龄 数值型 整数最近活跃天数 数值型 整数
# 练习1：用户画像的模拟与生成

### 用户画像生成
### 用户属性包含：
### 性别  男、女、未透露
### 所在城市： 北京、上海、广州、深圳、武汉、成都、重庆、杭州、厦门、其他
### 消费水平： 高、中、低
### 年龄：12-65
### 最近活跃天数：0-30

#### 数据格式 data = {'用户' ： 1, 性别：’nan‘， ’‘}

### 生成数据规模
import random
import csv

gender = ['男', '女', '未透露']
city = ['北京', '上海', '广州', '深圳', '武汉','成都', '重庆', '杭州','厦门', '其他']
consumption = ['高', '中', '低']

def generate_user_data():
    user_data = {}
    user_data['用户'] = None
    user_data['性别'] = random.choice(gender)
    user_data['所在城市'] = random.choice(city)
    user_data['消费水平'] = random.choice(consumption)
    user_data['年龄'] = random.randint(12, 65)
    user_data['最近活跃天数'] = random.randint(0, 30)
    return user_data

def generate_user_data_batch(num):
    user_data_list = []
    for i in range(num):
        user_data = generate_user_data()
        user_data['用户'] = i
        user_data_list.append(user_data)
    return user_data_list


if __name__ == '__main__':
    with open('user_data.csv', 'w', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['用户', '性别', '所在城市', '消费水平', '年龄', '最近活跃天数'])
        user_data_list = generate_user_data_batch(200)
        for user_data in user_data_list:
            writer.writerow(user_data.values())

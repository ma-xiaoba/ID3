# Auther:马占威
'''
                           _ooOoo_
                          o8888888o
                          88" . "88
                          (| -_- |)
                          O\  =  /O
                       ____/`---'\____
                     .'  \\|     |//  `.
                    /  \\|||  :  |||//  \
                   /  _||||| -:- |||||-  \
                   |   | \\\  -  /// |   |
                   | \_|  ''\---/''  |   |
                   \  .-\__  `-`  ___/-. /
                 ___`. .'  /--.--\  `. . __
              ."" '<  `.___\_<|>_/___.'  >'"".
             | | :  `- \`.;`\ _ /`;.`/ - ` : | |
             \  \ `-.   \_ __\ /__ _/   .-` /  /
        ======`-.____`-.___\_____/___.-`____.-'======
                           `=---='
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                 佛祖保佑       永无BUG
'''

# 手动实现决策树ID3算法
import numpy as np
import math
import pandas as pd
import pymysql


class Node:
    attribute_name = None
    attribute_value = None
    child_list = []
    class_y = None
    gain = None

    def toString(self):
        return "{ attribute_name: %s, attribute_value: %s, class_y: %s, gain: %s" %(self.attribute_name,self.attribute_value,self.class_y,self.gain)


class DecisionTree:
    root = Node()
    labels = []
    attribute_set = {}

    def __init__(self, labels, data):
        self.labels = labels
        self._get_data_set_(data)
        self._create_tree_(data, self.labels)
        print("-------init-------")

    def _get_data_set_(self, data):
        for label in self.labels:
            index = self.labels.index(label)
            column = [d[index] for d in data]
            self.attribute_set[label] = set(column)

    def _create_tree_(self, data, labels, node = None):
        if node is None:
            node = self.root
        class_y = [line[-1] for line in data]
        if len(set(class_y)) == 1:
            node.class_y = class_y[0]
            return
        if len(labels) == 0:
            node.class_y = self.majority_class(class_y)
            return
        # 从属性中找到最优划分属性(ID3算法)
        attribute_name_index, gain = self.find_best_attribute(data)
        node.attribute_name = labels[attribute_name_index]
        node.gain = gain
        # 最优属性的取值
        attribute_value_set = self.attribute_set[node.attribute_name]
        for attribute_value in attribute_value_set:
            child_node = Node()
            child_node.child_list = []
            child_node.attribute_value = attribute_value
            node.child_list.append(child_node)
            data_v = self.split_data_v(data, attribute_name_index, attribute_value)
            if data_v is None or len(data_v) == 0:
                class_v_y = [cls[-1] for cls in data]
                child_node.class_y = self.majority_class(class_v_y)
            else:
                labels_v = labels[:attribute_name_index] + labels[attribute_name_index + 1:]
                self._create_tree_(data_v, labels_v, child_node)
        return

    # 获取最优属性
    def find_best_attribute(self, data):
        # 属性的数量
        attribute_count = len(data[0]) - 1
        entropy = self.calculate_entropy(data)
        # 最大增益
        gain = 0
        best_attribute = -1
        for index in range(attribute_count):
            attributes = [a[index] for a in data]
            attribute_value_set = set(attributes)
            sum_ent_dv = 0
            for attribute_value in attribute_value_set:
                data_v = self.split_data_v(data, index, attribute_value)
                # 当前属性值出现的概率
                prob = len(data_v) / len(data)
                sum_ent_dv += prob * self.calculate_entropy(data_v)
            # 当前属性增益
            gain_v = entropy - sum_ent_dv
            if gain_v > gain:
                gain = gain_v
                best_attribute_index = index
        return [best_attribute_index, gain]

    # 计算熵
    def calculate_entropy(self, data):
        row_count = len(data)
        class_count = {}
        for row in data:
            clazz = row[-1]
            if clazz not in class_count.keys():
                class_count[clazz] = 0
            class_count[clazz] += 1
        entropy = 0.0
        for key in class_count.keys():
            value = class_count[key]
            entropy -= value / row_count * math.log2(value / row_count)
        return entropy

    # 属性划分
    def split_data_v(self, data, index, attribute_value):
        new_data = []
        for row in data:
            if row[index] == attribute_value:
                temp = row[:index] + row[index + 1:]
                new_data.append(temp)
        return new_data

    def majority_class(self, class_y):
        clazz_count = {}
        for clazz in class_y:
            if clazz not in clazz_count.keys():
                clazz_count[clazz] = 0
            clazz_count[clazz] += 1

        max = 0
        max_key = 0
        for clazz in clazz_count.keys():
            if clazz_count[clazz] > 0:
                max_key = clazz
                max = clazz_count[clazz]
        return max_key

    def print_tree(self):
        queue = []
        if self.root is not None:
            queue.append(self.root)
        while queue is not None and len(queue) > 0:
            pre_node = queue[0]
            print(pre_node.toString())
            for node in pre_node.child_list:
                queue.append(node)
            del queue[0]

    def predict(self, data, node=None):
        if node is None:
            node = self.root
        if node.class_y is not None:
            return node.class_y
        index = self.labels.index(node.attribute_name)
        attribute_value = data[index]
        if len(node.child_list) > 0:
            for child_node in node.child_list:
                if child_node.attribute_value == attribute_value:
                    return self.predict(data, child_node)


def getDataFromDB():
    # 打开数据库连接
    db = pymysql.connect("localhost", "root", "root", "ID3")
    # 使用 cursor() 方法创建一个游标对象 cursor
    cursor = db.cursor()
    # 使用 execute()  方法执行 SQL 查询
    sql = "SELECT colours,root,knock,texture,umbilicus,touch,class from melon"
    try:
        # 执行SQL语句
        cursor.execute(sql)
        # 获取所有记录列表
        data = cursor.fetchall()
    except:
        print("Error : unable to fetch data")
    db.close()
    return data


def main():
    data = getDataFromDB()
    labels = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感']
    tree = DecisionTree(labels, data)
    tree.print_tree()
    node = tree.predict(data=['青绿','蜷缩','沉闷','稍糊','稍凹','硬滑'])
    print(node)


# melon
if __name__ == "__main__":
    main()

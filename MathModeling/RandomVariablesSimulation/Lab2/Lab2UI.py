# Общие функции для вывода инф. на форму для Task1UI и Task2UI

from PyQt5 import QtGui



def print_table_to_textedit(headers, rows, textedit):
    cursor = textedit.textCursor()
    cursor.insertTable(len(rows) + 1, len(headers))
    for header in headers:
        cursor.insertText(header)
        cursor.movePosition(QtGui.QTextCursor.NextCell)
    for row in rows:
        for value in row:
            cursor.insertText(str(value))
            cursor.movePosition(QtGui.QTextCursor.NextCell)




def print_criterion_table(level, criterion, criterion_t):
    headers = ["   Уровень значимости   ", 
               " Практ. знач. критерия  ",
               "  Теор. знач. критерия  ",               
               "  Гипотеза принимается  "]
    rows = []
    for i in range(len(level)):
        conf_level = '             ' + str(level[i])
        crit = "     {:.6f}".format(criterion)
        crit_t = "     {:.6f}".format(criterion_t[i])
        result = '                Да' if criterion < criterion_t[i] else '                Нет'
        rows.append([conf_level, crit, crit_t, result ])
    
    return headers, rows



def print_point_estimates(pe_params, textedit):
    '''         Вывод точечных оценок
        pe_params - что выводить, textedit - куда  '''
    m, D, m_t, D_t, d_m, d_D = pe_params
    
    headers = ["", "    Точечное зн.   ", " Теоретическое зн. ", "        \u0394"]
    rows = [["    МО     ", '      %.6f'%m, '      %.6f'%m_t, '  %.6f  '%d_m],
            [" Дисперсия ", '      %.6f'%D, '      %.6f'%D_t, '  %.6f  '%d_D]]

    cursor = textedit.textCursor()
    cursor.insertText("                           Точечные оценки\n")
    print_table_to_textedit(headers, rows, textedit)


def print_interval_estimates(ie_params, pe_params, textedit):
    '''     Вывод интервальных оценок
        pe_params - параметры интервальных оценок, pe_params - параметры точечных оценок, textedit - куда '''
    m, D, m_t, D_t, d_m, d_D = pe_params

    M_ie_params, D_ie_params = ie_params
    level_m, eps, eps_D = M_ie_params
    level, l_border, r_border, l_border_M, r_border_M = D_ie_params

    cursor = textedit.textCursor()
    cursor.insertText("\n\n\n                                   Интервальные оценки\n")

    cursor.insertText("Доверительный интервал для оценки МО СВ при неизвестной дисперсии\n")
    headers, rows = print_m_table(level_m, m, m_t, eps)
    print_table_to_textedit(headers, rows, textedit)

    cursor.insertText("\n\nДоверительный интервал для оценки МО СВ при известной дисперсии\n")
    headers, rows = print_m_table(level_m, m, m_t, eps_D)
    print_table_to_textedit(headers, rows, textedit)

    cursor.insertText("\n\nДоверительный интервал для оценки дисперсии СВ при неизвестном МО\n")
    headers, rows = print_D_table(level, D, l_border, r_border, D_t)
    print_table_to_textedit(headers, rows, textedit)

    cursor.insertText("\n\nДоверительный интервал для оценки дисперсии СВ при известном МО\n")
    headers, rows = print_D_table(level, D, l_border_M, r_border_M, D_t)
    print_table_to_textedit(headers, rows, textedit)


def print_m_table(level_m, m, m_t, eps):
    headers = ["   Уровень значимости   ", 
               " Доверительный интервал ", 
               "    Накрывает теор. МО  "]
    rows = []
    for i in range(len(level_m)):
        conf_level = '             ' + str(level_m[i])
        interval = "        [{:.4f}, {:.4f}]".format(m-eps[i], m+eps[i])
        result = '                Да' if m-eps[i] <= m_t <= m+eps[i] else '                Нет'
        rows.append([conf_level, interval, result ])
    
    return headers, rows

def print_D_table(level, D, l_border, r_border, D_t):
    headers = ["   Уровень значимости   ", 
               " Доверительный интервал ", 
               "    Накрывает теор. D  "]
    rows = []
    for i in range(len(level)):
        conf_level = '             ' + str(level[i])
        interval = "        [{:.4f}, {:.4f}]".format(l_border[i], r_border[i])
        result = '                Да' if l_border[i] <= D_t <= r_border[i] else '                Нет'
        rows.append([conf_level, interval, result ])
    
    return headers, rows

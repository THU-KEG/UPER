import os
import xlwt
import xlrd
import argparse
import json


def set_style(name, height, bold=False):
    style = xlwt.XFStyle()

    font = xlwt.Font()
    font.name = name  # 'Times New Roman'
    font.bold = bold
    font.color_index = 4
    font.height = height
    style.font = font

    return style


def j2e_write(args):
    json_path = os.path.join(args.save_dir, "{}.json".format(args.category))
    fin = open(json_path, "r")
    text_dicts = json.load(fin)
    # print(len(text_dicts))
    fout = xlwt.Workbook()

    for i, text_dict in enumerate(text_dicts):
        row_id = 0
        sheet = fout.add_sheet('{}'.format(i), cell_overwrite_ok=True)
        for k, g_str in text_dict.items():
            sheet.write(row_id, 0, "##{}".format(k))
            row_id += 1
            sents = g_str.split("\n")
            for j, sent in enumerate(sents):
                sheet.write(row_id, 0, sent)
                row_id += 1
            # A blank line
            row_id += 1
        sheet.write(row_id, 0, "###If bart_tf_idf is better, fill the blank with 0:")
        row_id += 1
        for j in range(3):
            sheet.write(row_id, j, "annotator{}".format(j))

    fout.save(os.path.join(args.save_dir, '{}.xls'.format(args.category)))


def j2e_read(args):
    xls_path = os.path.join(args.save_dir, '{}.xls'.format(args.category))
    data = xlrd.open_workbook(xls_path)
    scores = []
    for table in data.sheets():
        row_id = 0
        while True:
            content = table.cell(row_id, 0).value
            if content.startswith('annotator'):
                a_id = row_id + 1
                all_annotators = table.cell(a_id, 0).value + table.cell(a_id, 1).value + table.cell(a_id, 2).value
                scores.append(all_annotators / 3.0)
                break
            row_id += 1

    print("Average score on {}:".format(args.category))
    print(sum(scores) / len(scores))


def main():
    parser = argparse.ArgumentParser()
    # parameters
    parser.add_argument('--category', default='animal', choices=['animal', 'film', 'company'])
    parser.add_argument('--task', default='w', choices=['w', 'r'])
    parser.add_argument('--save_dir', default='/home/tsq/TopCLS/human_evaluation')
    args = parser.parse_args()
    if args.task == 'w':
        j2e_write(args)
    else:
        j2e_read(args)


if __name__ == '__main__':
    main()

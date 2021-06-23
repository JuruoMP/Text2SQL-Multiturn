import os
import json
import attr
import copy
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from transformers import BartConfig, BartTokenizer
import networkx as nx
from seq2struct.datasets.spider_lib import evaluation


class OldSparcDataset(Dataset):
    def __init__(self, path, split='train'):
        self.path = path
        tables = json.load(open(os.path.join(self.path, 'tables.json'), 'r', encoding='utf-8'))
        self.tables = {x['db_id']: x for x in tables}
        raw_data = json.load(open(os.path.join(self.path, f'{split}.json'), 'r', encoding='utf-8'))
        self.data = []
        for example in raw_data:
            db_id = example['database_id']
            table_info = self.tables[db_id]
            accumulated_nl = []
            for interaction in example['interaction']:
                accumulated_nl.append(interaction['utterance'])
                triple = (copy.copy(accumulated_nl, interaction['query']), table_info)
                self.data.append(triple)
        self.bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

    def __getitem__(self, idx):
        nls, sql, table_info = self.data[idx]
        nl_ids = [self.bart_tokenizer(nl) for nl in nls]
        concat_nl_ids = [self.bart_tokenizer.sos_token_id]
        for nl_id in nl_ids:
            concat_nl_ids += nl_id + [self.bart_tokenizer.eos_token_id]

        sql_ids = self.bart_tokenizer(sql)

        table_names, column_names = table_info['table_names'], table_info['column_names']
        table_ids = [self.bart_tokenizer(x) for x in table_names]
        column_ids = [self.bart_tokenizer(x[1]) for x in column_names]
        concat_table_ids = [0]

        concat_input_ids = torch.Tensor(concat_nl_ids + concat_table_ids)
        attention_mask = torch.ones_like(concat_input_ids)
        output_ids = torch.Tensor(sql_ids)
        return {
            'input_ids': concat_input_ids,
            'attention_mask': attention_mask,
            'labels': output_ids
        }

    def collate_fn(self, data_list):
        raise NotImplementedError


@attr.s
class SparcItem:
    text = attr.ib()
    code = attr.ib()
    schema = attr.ib()
    orig = attr.ib()
    orig_schema = attr.ib()


@attr.s
class Column:
    id = attr.ib()
    table = attr.ib()
    name = attr.ib()
    unsplit_name = attr.ib()
    orig_name = attr.ib()
    type = attr.ib()
    foreign_key_for = attr.ib(default=None)


@attr.s
class Table:
    id = attr.ib()
    name = attr.ib()
    unsplit_name = attr.ib()
    orig_name = attr.ib()
    columns = attr.ib(factory=list)
    primary_keys = attr.ib(factory=list)


@attr.s
class Schema:
    db_id = attr.ib()
    tables = attr.ib()
    columns = attr.ib()
    foreign_key_graph = attr.ib()
    orig = attr.ib()


def load_tables(path):
    schemas = {}
    eval_foreign_key_maps = {}

    schema_dicts = json.load(open(path))
    for schema_dict in schema_dicts:
        tables = tuple(
            Table(
                id=i,
                name=name.split(),
                unsplit_name=name,
                orig_name=orig_name,
            )
            for i, (name, orig_name) in enumerate(zip(
                schema_dict['table_names'], schema_dict['table_names_original']))
        )
        columns = tuple(
            Column(
                id=i,
                table=tables[table_id] if table_id >= 0 else None,
                name=col_name.split(),
                unsplit_name=col_name,
                orig_name=orig_col_name,
                type=col_type,
            )
            for i, ((table_id, col_name), (_, orig_col_name), col_type) in enumerate(zip(
                schema_dict['column_names'],
                schema_dict['column_names_original'],
                schema_dict['column_types']))
        )

        # Link columns to tables
        for column in columns:
            if column.table:
                column.table.columns.append(column)

        for column_id in schema_dict['primary_keys']:
            # Register primary keys
            column = columns[column_id]
            column.table.primary_keys.append(column)

        foreign_key_graph = nx.DiGraph()
        for source_column_id, dest_column_id in schema_dict['foreign_keys']:
            # Register foreign keys
            source_column = columns[source_column_id]
            dest_column = columns[dest_column_id]
            source_column.foreign_key_for = dest_column
            foreign_key_graph.add_edge(
                source_column.table.id,
                dest_column.table.id,
                columns=(source_column_id, dest_column_id))
            foreign_key_graph.add_edge(
                dest_column.table.id,
                source_column.table.id,
                columns=(dest_column_id, source_column_id))

        db_id = schema_dict['db_id']
        assert db_id not in schemas
        schemas[db_id] = Schema(db_id, tables, columns, foreign_key_graph, schema_dict)
        eval_foreign_key_maps[db_id] = evaluation.build_foreign_key_map(schema_dict)

    return schemas, eval_foreign_key_maps


class SparcDataset(torch.utils.data.Dataset):
    def __init__(self, path, tables_paths, db_path, limit=None):
        self.path = path
        self.db_path = db_path
        self.examples = []
        self.use_column_type = False
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
        self.max_seq_len = self.tokenizer.model_max_length

        self.schemas, self.eval_foreign_key_maps = load_tables(tables_paths)

        raw_data = json.load(open(path))
        for entry in tqdm(raw_data):
            accumulated_toks = []
            for i, interaction in enumerate(entry['interaction']):
                new_toks = interaction['utterance_toks']
                accumulated_toks.append(new_toks)
                item = SparcItem(
                    text=copy.deepcopy(accumulated_toks),
                    code=interaction['query'],
                    schema=self.schemas[entry['database_id']],
                    orig=(entry, i),
                    orig_schema=self.schemas[entry['database_id']].orig)
                if self.validate_item(item):
                    self.examples.append(item)

        print('Sparc dataset built.')

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        encoder_dict, decoder_dict = self.tokenize_item(item)
        return {
            'input_ids': encoder_dict['input_ids'],
            'attention_mask': encoder_dict['attention_mask'],
            'decoder_input_ids': decoder_dict['input_ids'],
            'decoder_attention_mask': decoder_dict['attention_mask']
        }

    def tokenize_item(self, item):
        nl = ' '.join([t for s in item.text for t in s])
        sql = item.code
        columns = []
        for c in item.schema.columns:
            if c and c.table:
                tn, cn = c.table.orig_name, c.orig_name
                columns.append((tn, cn))
        concat_input = nl + self.tokenizer.eos_token
        for c in columns:
            concat_input += c[0] + ' ' + c[1] + self.tokenizer.eos_token
        concat_input.rstrip(self.tokenizer.eos_token)
        encoder_dict = self.tokenizer(concat_input)
        decoder_dict = self.tokenizer(sql)
        decoder_dict['input_ids'][0] = self.tokenizer.eos_token_id
        return encoder_dict, decoder_dict

    def validate_item(self, item):
        encoder_dict, decoder_dict = self.tokenize_item(item)
        return len(encoder_dict['input_ids']) < self.max_seq_len and len(decoder_dict['input_ids']) < self.max_seq_len

    def collate_fn(self, x_list):
        max_input_len = max(len(x['input_ids']) for x in x_list)
        max_output_len = max(len(x['decoder_input_ids']) for x in x_list)
        for x in x_list:
            x['input_ids'] += [0 for _ in range(max_input_len - len(x['input_ids']))]
            x['attention_mask'] += [0 for _ in range(max_input_len - len(x['attention_mask']))]
            x['decoder_input_ids'] += [-100 for _ in range(max_output_len - len(x['decoder_input_ids']))]
            x['decoder_attention_mask'] += [0 for _ in range(max_input_len - len(x['decoder_attention_mask']))]
        return default_collate([{k: torch.tensor(v).long() for k, v in x.items()} for x in x_list])

    class Metrics:
        def __init__(self, dataset):
            self.dataset = dataset
            self.foreign_key_maps = {
                db_id: evaluation.build_foreign_key_map(schema.orig)
                for db_id, schema in self.dataset.schemas.items()
            }
            self.evaluator = evaluation.Evaluator(
                self.dataset.db_path,
                self.foreign_key_maps,
                'match')
            self.results = []

        def add(self, item, inferred_code, orig_question=None):
            ret_dict = self.evaluator.evaluate_one(
                item.schema.db_id, item.orig['query'], inferred_code)
            if orig_question:
                ret_dict["orig_question"] = orig_question
            self.results.append(ret_dict)

        def add_beams(self, item, inferred_codes, orig_question=None):
            beam_dict = {}
            if orig_question:
                beam_dict["orig_question"] = orig_question
            for i, code in enumerate(inferred_codes):
                ret_dict = self.evaluator.evaluate_one(
                    item.schema.db_id, item.orig['query'], code)
                beam_dict[i] = ret_dict
                if ret_dict["exact"] is True:
                    break
            self.results.append(beam_dict)

        def finalize(self):
            self.evaluator.finalize()
            return {
                'per_item': self.results,
                'total_scores': self.evaluator.scores
            }


# class SparcEncoderBartPreproc(SparcEncoderV2Preproc):
#     # Why in the BERT model, we set the include_table_name_in_column as False?
#     def __init__(
#             self,
#             save_path,
#             db_path,
#             fix_issue_16_primary_keys=False,
#             include_table_name_in_column=False,
#             bart_version = "bart-large",
#             compute_sc_link=True,
#             compute_cv_link=False):
#         self.data_dir = os.path.join(save_path, 'enc')
#         self.db_path = db_path
#         self.texts = collections.defaultdict(list)
#         self.fix_issue_16_primary_keys = fix_issue_16_primary_keys
#         self.include_table_name_in_column = include_table_name_in_column
#         self.compute_sc_link = compute_sc_link
#         self.compute_cv_link = compute_cv_link
#
#         self.counted_db_ids = set()
#         self.preprocessed_schemas = {}
#
#         self.tokenizer = BartTokenizer.from_pretrained(bart_version)
#
#         column_types = ["text", "number", "time", "boolean", "others"]
#         self.tokenizer.add_tokens([f"<type: {t}>" for t in column_types])
#
#     # def _tokenize(self, presplit, unsplit):
#     #     # I want to keep this tokenization consistent with BartTokens.
#     #     # Presplit is required here.
#     #     tokens = nltk.word_tokenize(unsplit.replace("'", " ' ").replace('"', ' " '))
#     #     toks = []
#     #     for token in tokens:
#     #         toks.extend(self.tokenizer.tokenize(token, add_prefix_space=True))
#     #     return toks
#     def _tokenize(self, presplit, unsplit):
#         s = ' '.join(presplit)
#         toks = self.tokenizer.tokenize(s)
#         return toks
#
#     def add_item(self, item, section, validation_info):
#         preprocessed = self.preprocess_item(item, validation_info)
#         self.texts[section].append(preprocessed)
#
#     def preprocess_item(self, item, validation_info):
#         # For bart, there is a punctuation issue if we want to merge it back to words.
#         # So here I will use nltk to further tokenize the sentence first.
#         turn_texts = [' '.join(x) for x in item.text]
#         concat_turn_texts = ' </s> '.join(turn_texts)
#         question = self._tokenize(concat_turn_texts.split(' '), concat_turn_texts)
#         preproc_schema = self._preprocess_schema(item.schema)
#         question_bart_tokens = BartTokens(concat_turn_texts, self.tokenizer)
#         if self.compute_sc_link:
#             # We do not want to transform pieces back to word.
#             sc_link = question_bart_tokens.bart_schema_linking(
#                 preproc_schema.normalized_column_names,
#                 preproc_schema.normalized_table_names
#             )
#         else:
#             sc_link = {"q_col_match": {}, "q_tab_match": {}}
#
#         if self.compute_cv_link:
#             cv_link = question_bart_tokens.bart_cv_linking(
#                 item.schema, self.db_path)
#         else:
#             cv_link = {"num_date_match": {}, "cell_match": {}}
#
#         return {
#             'raw_question': concat_turn_texts,
#             'question': question,
#             'db_id': item.schema.db_id,
#             'sc_link': sc_link,
#             'cv_link': cv_link,
#             'columns': preproc_schema.column_names,
#             'tables': preproc_schema.table_names,
#             'table_bounds': preproc_schema.table_bounds,
#             'column_to_table': preproc_schema.column_to_table,
#             'table_to_columns': preproc_schema.table_to_columns,
#             'foreign_keys': preproc_schema.foreign_keys,
#             'foreign_keys_tables': preproc_schema.foreign_keys_tables,
#             'primary_keys': preproc_schema.primary_keys,
#         }
#
#     def validate_item(self, item, section):
#         turn_texts = [' '.join(x) for x in item.text]
#         concat_turn_texts = ' </s> '.join(turn_texts)
#         question = self._tokenize(concat_turn_texts.split(' '), concat_turn_texts)
#         preproc_schema = self._preprocess_schema(item.schema)
#         # 2 is for cls and sep special tokens. +1 is for sep
#         num_words = len(question) + \
#                     sum(len(c) + 1 for c in preproc_schema.column_names) + \
#                     sum(len(t) + 1 for t in preproc_schema.table_names)
#         if num_words > 512:
#             return False, None  # remove long sequences
#         else:
#             return True, None
#
#     def _preprocess_schema(self, schema):
#         if schema.db_id in self.preprocessed_schemas:
#             return self.preprocessed_schemas[schema.db_id]
#         result = preprocess_schema_uncached_bart(schema, self.tokenizer, self._tokenize,
#                                             self.include_table_name_in_column,
#                                             self.fix_issue_16_primary_keys, bart=True)
#         self.preprocessed_schemas[schema.db_id] = result
#         return result
#
#     def save(self):
#         os.makedirs(self.data_dir, exist_ok=True)
#         self.tokenizer.save_pretrained(self.data_dir)
#
#         for section, texts in self.texts.items():
#             with open(os.path.join(self.data_dir, section + '.jsonl'), 'w') as f:
#                 for text in texts:
#                     f.write(json.dumps(text) + '\n')
#
#     def load(self):
#         self.tokenizer = BartTokenizer.from_pretrained(self.data_dir)


if __name__ == '__main__':
    train_data = SparcDataset('sparc/train.json', 'sparc/tables.json', 'sparc/database')
    dataloader = torch.utils.data.DataLoader(train_data, batch_size=7, collate_fn=train_data.collate_fn)
    for batch in dataloader:
        a = 1
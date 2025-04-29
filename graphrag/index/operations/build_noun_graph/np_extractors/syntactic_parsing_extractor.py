# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Noun phrase extractor based on dependency parsing and NER using SpaCy."""

from typing import Any

from spacy.tokens.span import Span
from spacy.util import filter_spans

from graphrag.index.operations.build_noun_graph.np_extractors.base import (
    BaseNounPhraseExtractor,
)
from graphrag.index.operations.build_noun_graph.np_extractors.np_validator import (
    has_valid_token_length,
    is_compound,
    is_valid_entity,
)


class SyntacticNounPhraseExtractor(BaseNounPhraseExtractor):
    """Noun phrase extractor based on dependency parsing and NER using SpaCy."""

    def __init__(
        self,
        model_name: str,
        max_word_length: int,
        include_named_entities: bool,
        exclude_entity_tags: list[str],
        exclude_pos_tags: list[str],
        exclude_nouns: list[str],
        word_delimiter: str,
    ):
        """
        Noun phrase extractor based on dependency parsing and NER using SpaCy.

        This extractor tends to produce more accurate results than regex-based extractors but is slower.
        Also, it can be used for different languages by using the corresponding models from SpaCy.

        Args:
            model_name: SpaCy model name.
            max_word_length: Maximum length (in character) of each extracted word.
            include_named_entities: Whether to include named entities in noun phrases
            exclude_entity_tags: list of named entity tags to exclude in noun phrases.
            exclude_pos_tags: List of POS tags to remove in noun phrases.
            word_delimiter: Delimiter for joining words.
        """
        super().__init__(
            model_name=model_name,
            max_word_length=max_word_length,
            exclude_nouns=exclude_nouns,
            word_delimiter=word_delimiter,
        )
        self.include_named_entities = include_named_entities
        self.exclude_entity_tags = exclude_entity_tags
        if not include_named_entities:
            self.nlp = self.load_spacy_model(model_name, exclude=["lemmatizer", "ner"])
        else:
            self.nlp = self.load_spacy_model(model_name, exclude=["lemmatizer"])

        self.exclude_pos_tags = exclude_pos_tags

    #! EXTRACT FOR CHINESE TEXT
    def extract(
        self,
        text: str,
    ) -> list[str]:
        """
        从文本中提取名词短语（适配中文，替换原生 noun_chunks 逻辑）
        """
        doc = self.nlp(text)
        
        # 自定义中文名词短语提取函数
        def extract_cn_noun_chunks(doc):
            chunks = []
            # 规则1: 基于依存关系提取核心名词及其修饰词
            for token in doc:
                if token.pos_ in ["NOUN", "PROPN"] and token.dep_ in ["nsubj", "dobj", "ROOT"]:
                    subtree = [t for t in token.subtree]
                    start = subtree[0].i
                    end = subtree[-1].i + 1
                    chunks.append(doc[start:end])
            
            # 规则2: 使用 Matcher 匹配量词+名词结构（例如"三只小鸟"）
            from spacy.matcher import Matcher
            matcher = Matcher(self.nlp.vocab)
            patterns = [
                [{"POS": "NUM"}, {"POS": "NOUN"}],  # 数字 + 名词
                [{"POS": "ADJ"}, {"POS": "NOUN"}]   # 形容词 + 名词
            ]
            patterns += [
                [{"POS": "NOUN"}, {"POS": "NOUN"}],  # 名词+名词（例如"学生宿舍"）
                [{"POS": "ADJ"}, {"POS": "ADJ"}, {"POS": "NOUN"}]  # 多形容词+名词
            ]
            matcher.add("CN_NOUN_PHRASES", patterns)
            # 专门匹配 "数量词 + 名词"（例如"一群学生"）
            matcher.add("QUANTIFIER", [[
                {"POS": "NUM"},
                {"POS": "NOUN", "DEP": "clf"},  # clf 表示量词
                {"POS": "NOUN"}
            ]])
            matches = matcher(doc)
            for _, start, end in matches:
                chunks.append(doc[start:end])
            
            # 去重并返回
            return list({span.text: span for span in chunks}.values())

        filtered_noun_phrases = set()
        entities = [ent for ent in doc.ents if ent.label_ not in self.exclude_entity_tags] if self.include_named_entities else []
        
        # 替换原生 noun_chunks 为自定义方法
        custom_noun_chunks = extract_cn_noun_chunks(doc)  # 自定义提取
        
        spans = entities + custom_noun_chunks  # 原 doc.noun_chunks 替换为 custom_noun_chunks
        spans = filter_spans(spans)
        
        # 剩余逻辑保持不变...
        missing_entities = [
            ent for ent in entities
            if not any(ent.text in span.text for span in spans)
        ]
        spans.extend(missing_entities)

        tagged_noun_phrases = [
            self._tag_noun_phrases(span, entities) for span in spans
        ]
        for tagged_np in tagged_noun_phrases:
            if (tagged_np["is_valid_entity"]) or (
                (
                    len(tagged_np["cleaned_tokens"]) > 1
                    or tagged_np["has_compound_words"]
                )
                and tagged_np["has_valid_tokens"]
            ):
                filtered_noun_phrases.add(tagged_np["cleaned_text"])

        return list(filtered_noun_phrases)
    # def extract(
    #     self,
    #     text: str,
    # ) -> list[str]:
    #     """
    #     Extract noun phrases from text. Noun phrases may include named entities and noun chunks, which are filtered based on some heuristics.

    #     Args:
    #         text: Text.

    #     Returns: List of noun phrases.
    #     """
    #     doc = self.nlp(text)

    #     filtered_noun_phrases = set()
    #     if self.include_named_entities:
    #         # extract noun chunks + entities then filter overlapping spans
    #         entities = [
    #             ent for ent in doc.ents if ent.label_ not in self.exclude_entity_tags
    #         ]
    #         spans = entities + list(doc.noun_chunks)
    #         spans = filter_spans(spans)

    #         # reading missing entities
    #         missing_entities = [
    #             ent
    #             for ent in entities
    #             if not any(ent.text in span.text for span in spans)
    #         ]
    #         spans.extend(missing_entities)

    #         # filtering noun phrases based on some heuristics
    #         tagged_noun_phrases = [
    #             self._tag_noun_phrases(span, entities) for span in spans
    #         ]
    #         for tagged_np in tagged_noun_phrases:
    #             if (tagged_np["is_valid_entity"]) or (
    #                 (
    #                     len(tagged_np["cleaned_tokens"]) > 1
    #                     or tagged_np["has_compound_words"]
    #                 )
    #                 and tagged_np["has_valid_tokens"]
    #             ):
    #                 filtered_noun_phrases.add(tagged_np["cleaned_text"])
    #     else:
    #         tagged_noun_phrases = [
    #             self._tag_noun_phrases(chunk, []) for chunk in doc.noun_chunks
    #         ]
    #         for tagged_np in tagged_noun_phrases:
    #             if (tagged_np["has_proper_noun"]) or (
    #                 (
    #                     len(tagged_np["cleaned_tokens"]) > 1
    #                     or tagged_np["has_compound_words"]
    #                 )
    #                 and tagged_np["has_valid_tokens"]
    #             ):
    #                 filtered_noun_phrases.add(tagged_np["cleaned_text"])

    #     return list(filtered_noun_phrases)

    def _tag_noun_phrases(
        self, noun_chunk: Span, entities: list[Span]
    ) -> dict[str, Any]:
        """Extract attributes of a noun chunk, to be used for filtering."""
        cleaned_tokens = [
            token
            for token in noun_chunk
            if token.pos_ not in self.exclude_pos_tags
            and token.text.upper() not in self.exclude_nouns
            and token.is_space is False
            and not token.is_punct
        ]
        cleaned_token_texts = [token.text for token in cleaned_tokens]
        cleaned_text_string = (
            self.word_delimiter.join(cleaned_token_texts).replace("\n", "").upper()
        )

        noun_chunk_entity_labels = [
            (ent.text, ent.label_) for ent in entities if noun_chunk.text == ent.text
        ]
        if noun_chunk_entity_labels:
            noun_chunk_entity_label = noun_chunk_entity_labels[0]
            valid_entity = is_valid_entity(noun_chunk_entity_label, cleaned_token_texts)
        else:
            valid_entity = False

        return {
            "cleaned_tokens": cleaned_tokens,
            "cleaned_text": cleaned_text_string,
            "is_valid_entity": valid_entity,
            "has_proper_nouns": any(token.pos_ == "PROPN" for token in cleaned_tokens),
            "has_compound_words": is_compound(cleaned_token_texts),
            "has_valid_tokens": has_valid_token_length(
                cleaned_token_texts, self.max_word_length
            ),
        }

    def __str__(self) -> str:
        """Return string representation of the extractor, used for cache key generation."""
        return f"syntactic_{self.model_name}_{self.max_word_length}_{self.include_named_entities}_{self.exclude_entity_tags}_{self.exclude_pos_tags}_{self.exclude_nouns}_{self.word_delimiter}"

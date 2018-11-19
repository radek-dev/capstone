SHOW DATABASES;

DROP DATABASE IF EXISTS capstone;
CREATE DATABASE capstone;

SHOW TABLES;

DROP TABLE IF EXISTS bid;

create table bid
(
  `index`   text     null,
  price     double   null,
  amount    double   null,
  updatedId bigint   null,
  myUtc     datetime null,
  symbol    text     null
);


DROP TABLE IF EXISTS ask;

create table ask
(
  `index`   bigint   null,
  price     double   null,
  amount    double   null,
  updatedId bigint   null,
  myUtc     datetime null,
  symbol    text     null
);

create index ix_ask_index
  on ask (`index`);


DROP TABLE IF EXISTS symbols;


create table symbols
(
  `index` bigint null,
  price   text   null,
  symbol  text   null
);

create index ix_symbols_index
  on symbols (`index`);



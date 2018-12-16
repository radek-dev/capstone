# some queries to get relevant data from the database

# --------------bid prices ------------------------------
select b.price, sum(b.amount) as amount
from bid b
  inner join symbolRef sR on b.symbolId = sR.id
where b.updatedId = 50106062
group by b.price;


select b.price, b.amount, b.updatedId, b.myUtc, sR.symbol
    from bid b
inner join symbolRef sR on b.symbolId = sR.id
where b.updatedId = 50106062
order by b.myUtc, b.updatedId, b.price;


select distinct bid.updatedId
from bid
order by bid.updatedId asc
limit 60;

select min(bid.updatedId) from bid;


 select b.price, sum(b.amount) as amount
  from bid b
    inner join symbolRef sR on b.symbolId = sR.id
  where b.updatedId = 50106062
      and b.price >= 0.0002245
  group by b.price
  order by b.price DESC;


# --------------ask prices ------------------------------
select a.price, a.amount, a.updatedId, a.myUtc, sR.symbol
from ask a
  inner join symbolRef sR on a.symbolId = sR.id
order by a.myUtc, a.updatedId;


select a.price, sum(a.amount) as amount
from ask a
  inner join symbolRef sR on a.symbolId = sR.id
where a.updatedId = 50106062
group by a.price;


# ---------------bid and ask ----------------------------
# this query is slow
select max(b.price) as maxBid, min(a.price) as minAsk
from bid b inner join ask a
  on b.updatedId = a.updatedId
group by b.updatedId;

select qryMinAsk.updatedId, qryMaxBid.bidPrice, qryMinAsk.askPrice from
  (select b.updatedId, max(b.price) as bidPrice
  from bid b
  group by b.updatedId) as qryMaxBid
inner join
  (select a.updatedId, min(a.price) as askPrice
  from ask a
  group by a.updatedId) as qryMinAsk
on qryMinAsk.updatedId = qryMaxBid.updatedId order by qryMinAsk.updatedId


 select distinct bid.updatedId
        from bid
        where DATE(myUtc) = '2018-11-22'
        order by bid.updatedId asc
        limit 60;


select * from ask
where DATE(myUtc) = '2018-11-22'
and price < 0.00027;

select * from ask
where DATE(myUtc) = '2018-11-22'
and price > 0.00027;

select distinct DATE(myUtc) from ask;


 select qryMinAsk.updatedId, qryMaxBid.bidPrice, qryMinAsk.askPrice from
      (select b.updatedId, max(b.price) as bidPrice
      from bid b
      where DATE(myUtc) = '2018-12-10'
      group by b.updatedId) as qryMaxBid
    inner join
      (select a.updatedId, min(a.price) as askPrice
      from ask a
      where DATE(myUtc) = '2018-12-10'
      group by a.updatedId
      ) as qryMinAsk
    on qryMinAsk.updatedId = qryMaxBid.updatedId order by qryMinAsk.updatedId;


select b.updatedId, max(b.price), b.myUtc as bidPrice
      from bid b
      where DATE(myUtc) = '2018-11-22'
      group by b.updatedId;
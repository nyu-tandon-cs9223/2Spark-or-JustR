package ScalaProject

import org.apache.spark.rdd.RDD

class Stats(val tp: Double, val tn: Double, val fp: Double, val fn: Double) {

  /**
    * accuracy
    */
  val ACC = (tp + tn) / (tp + fp + fn + tn)
  val accuracy = ACC


  override def toString = {
      s"ACC (accuracy): $ACC \n"
  }
}

object Stats {

  def apply(cc: (Int, Int, Int, Int)): Stats = new Stats(cc._1, cc._2, cc._3, cc._4)

  def confusionMatrix(rdd: RDD[(Double, Double)]) = {
    rdd.aggregate((0, 0, 0, 0))(
      seqOp = (t, pal) => {
        val (tp, tn, fp, fn) = t
        (if (pal._1 == pal._2 && pal._2 == 1.0) tp + 1 else tp,
          if (pal._1 == pal._2 && pal._2 == 0.0) tn + 1 else tn,
          if (pal._1 == 1.0 && pal._2 == 0.0) fp + 1 else fp,
          if (pal._1 == 0.0 && pal._2 == 1.0) fn + 1 else fn)
      },
      combOp = (t1, t2) => t1 + t2)
  }

  implicit class Tupple4Add[A: Numeric, B: Numeric, C: Numeric, D: Numeric](t: (A, B, C, D)) {
    import Numeric.Implicits._
    def +(p: (A, B, C, D)) = (p._1 + t._1, p._2 + t._2, p._3 + t._3, p._4 + t._4)
  }

}
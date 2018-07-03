package ScalaProject
/**
  * Created by Shweta on 12/7/2017.
  */
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vector

trait LabeledPointConverter {
  def label(): Double
  def features(): Vector
  def toLabeledPoint() = LabeledPoint(label(), features())
}

case class TitanicSurvivorModel(passengerId: Int,
                          survived: Int,
                          pClass: Int,
                          sex: String,
                          age: Float,
                          sibSp: Int,
                          parCh: Int,
                          fare: Float,
                          embarked: String)
  extends LabeledPointConverter {

  def label() = survived.toDouble
  def features() = TitanicSurvivorModel.convert(this)
}

object TitanicSurvivorModel {

  def apply(row: Array[String]) = new TitanicSurvivorModel(
    row(0).toInt, row(1).toInt, row(2).toInt, row(3),
    row(4).replaceFirst(",", ".").toFloat, row(5).toInt,
    row(6).toInt, row(7).replaceFirst(",", ".").toFloat,
    row(8))

  def categoricalFeaturesInfo() = {
    Map[Int, Int](2 -> 2, 7 -> 3)
  }

  def convert(model: TitanicSurvivorModel) = Vectors.dense(
    model.passengerId.toDouble,
    model.pClass.toDouble,
    model.sex match {
      case "male" => 0d
      case "female" => 1d
    },
    model.age.toDouble,
    model.sibSp.toDouble,
    model.parCh.toDouble,
    model.fare,
    model.embarked match {
      case "S"  => 0d
      case "C"  => 1d
      case "Q"  => 2d
    }
  )
}
